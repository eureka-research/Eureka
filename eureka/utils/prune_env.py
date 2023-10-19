import os
import yaml
import ast
import sys

def modify_python_file(task, filename, output):
    # Create base environments compatible with Eureka generated rewards
    with open(filename, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    inside_docstring = False
    prev_line = None

    copied_lines = []
    keep_copying = False 
    in_success = False 

    for line in lines:
        stripped = line.strip()
        # Ignore comment lines
        if stripped.startswith("#"):
            if not stripped.startswith("# type:"):
                continue
        
        # Check for docstrings (triple-quoted strings)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not inside_docstring:
                inside_docstring = True
            else:
                inside_docstring = False
            continue
        
        # Ignore lines inside docstrings
        if inside_docstring:
            continue
        
        if f"compute_{task}_reward" in line:

            if "@torch.jit.script" not in prev_line:
                # Create compute_success function 
                line = line.replace("self.rew_buf[:], ", "self.gt_rew_buf, ")
                line = line.replace(f"compute_{task}_reward", "compute_success")
                in_success = True 
            else:
                # Replace "compute_hand_reward" with "compute_success" in the current line
                line = line.replace(f"compute_{task}_reward", "compute_success")
            modified_lines.append(line)
        else:
            if in_success and line.strip() == ")":
                modified_lines.append(line)
                modified_lines.append("        self.extras['gt_reward'] = self.gt_rew_buf.mean()")
                in_success = False
            else:    
                modified_lines.append(line)
        prev_line = line

    # Write the modified lines back to the file
    with open(output, 'w') as file:
        file.writelines(modified_lines)

def prune_python_class(filename, output, methods_to_keep, new_docstring, methods_to_prune_docstring):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Find the first line where a class is defined
    class_line = next(i for i, line in enumerate(lines) if line.strip().startswith('class'))

    # Keep lines from the class definition onwards
    lines = lines[class_line:]

    # Remove methods not in methods_to_keep
    pruned_lines = []
    inside_method_to_keep = False
    inside_docstring_to_prune = False
    docstring_started = False
    docstring_replaced = False
    for line in lines:
        if line.count('#') > 5:
            # Stop processing if a line with more than 5 "#" is encountered
            break
        stripped = line.strip()
        if stripped.startswith('class') and not docstring_replaced:
            # This is the class definition, append the new docstring after it
            pruned_lines.append(line)
            pruned_lines.append('    """' + new_docstring + '"""\n')
            docstring_replaced = True
        elif stripped.startswith('def') and not stripped.startswith('default'):
            # This is a method definition
            method_name = stripped.split('(')[0][4:]  # Get the method name
            inside_method_to_keep = method_name in methods_to_keep
            inside_docstring_to_prune = method_name in methods_to_prune_docstring
            if inside_method_to_keep:
                pruned_lines.append(line)
        else:
            if inside_method_to_keep and inside_docstring_to_prune:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if docstring_started:
                        # End of docstring
                        docstring_started = False
                        inside_docstring_to_prune = False
                    else:
                        # Start of docstring
                        docstring_started = True
                elif not docstring_started:
                    pruned_lines.append(line)
            elif inside_method_to_keep:
                pruned_lines.append(line)

    # Write the pruned lines back to the file
    with open(output, 'w') as file:
        file.writelines(pruned_lines[:-1])

def prune_reward(filename, output, method_to_keep):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Remove methods not in methods_to_keep
    pruned_lines = []
    inside_method_to_keep = False
    seen_jit = False
    prev_line = None
    for line in lines:
        stripped = line.strip()
        if method_to_keep in line:
            if "@torch.jit.script" in prev_line:
                pruned_lines.append(prev_line)
                pruned_lines.append(line)
                inside_method_to_keep = True
                continue
        else: 
            if inside_method_to_keep == True:
                if stripped == '"""':
                    break
                pruned_lines.append(line)
                if stripped == '"""':
                    break
        prev_line = line

    # Write the pruned lines back to the file
    with open(output, 'w') as file:
        file.writelines(pruned_lines)

## Creating YAML files for GPT prompting ##
def extract_class_info(file_path):
    with open(file_path, "r") as source:
        tree = ast.parse(source.read())
    
    class_node = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)][0]
    class_name = class_node.name
    class_docstring = ast.get_docstring(class_node)

    # Get first paragraph of docstring
    first_paragraph = class_docstring.split('\n\n')[0]
    
    return class_name, first_paragraph

def create_yaml(file_path, yaml_path):
    class_name, description = extract_class_info(file_path)
    env_name = os.path.splitext(os.path.basename(file_path))[0]
    
    description = description.replace('\n', ' ')
    
    data = {
        'task': class_name,
        'env_name': env_name,
        'description': description
    }

    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:  
        task = sys.argv[1]
        if len(sys.argv) > 2:
            reward_name = sys.argv[2]
        else:
            reward_name = task
    else:
        print("Usage: python prune_env_isaac.py <task> <optional: reward_name>")

    CURRENT_DIR = os.getcwd()
    ISAAC_ROOT_DIR = f"{CURRENT_DIR}/../../isaacgymenvs/isaacgymenvs"

    # Create base environment file to write reward function for
    modify_python_file(reward_name, f"{ISAAC_ROOT_DIR}/tasks/{task}.py", f"../envs/isaac/{task}.py")
    
    # Create a condensed version to serve as input to Eureka
    prune_python_class(f"../envs/isaac/{task}.py", f"../envs/isaac/{task}_obs.py", 
                    ["compute_observations", f"compute_{task}_observations", "_update_states",
                        f"compute_{task}_observations_states", "compute_full_state"],
                        
                        "Rest of the environment definition omitted.", [f"compute_{task}_reward"])