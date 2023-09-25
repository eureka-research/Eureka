import yaml

# # Load the YAML file
# task = 'Cartpole'
# suffix = 'GPT'

def create_task(root_dir, task, env_name, suffix):
    # Create task YAML file 
    input_file = f"{root_dir}/cfg/task/{task}.yaml"
    output_file = f"{root_dir}/cfg/task/{task}{suffix}.yaml"
    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['name'] = f'{task}{suffix}'
    data['env']['env_name'] = f'{env_name}{suffix}'
    
    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

    # Create training YAML file
    input_file = f"{root_dir}/cfg/train/{task}PPO.yaml"
    output_file = f"{root_dir}/cfg/train/{task}{suffix}PPO.yaml"

    with open(input_file, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)

    # Modify the "name" field
    data['params']['config']['name'] = data['params']['config']['name'].replace(task, f'{task}{suffix}')

    # Write the new YAML file
    with open(output_file, 'w') as new_yamlfile:
        yaml.safe_dump(data, new_yamlfile)

    # # Add the new task to the task map
    # task_map_file = f"{root_dir}/tasks/__init__.py"
    # # Open the file and read its contents
    # with open(task_map_file, 'r') as f:
    #     lines = f.readlines()

    # # Find the last import line (from the top)
    # import_maps = set()
    # last_import_line = 0
    # for i, line in enumerate(lines):
    #     if line.startswith('from .'):
    #         last_import_line = i
    #         import_maps.add(line)
    # # last_import_line = max(i for i, line in enumerate(lines) if line.startswith('from .')) 

    # import_insert = f'from .{env_name.lower()}{suffix.lower()} import {task}{suffix}\n'
    # if import_insert not in import_maps:
    #     # Add the new import line after the last one
    #     lines.insert(last_import_line + 1, import_insert)

    # # Find the line where the dictionary starts (from the bottom)
    # dict_start_line = len(lines) - next(i for i, line in reversed(list(enumerate(lines))) if 'isaacgym_task_map' in line) - 1

    # # Find the line where the dictionary ends
    # task_maps = set()
    # for i, line in enumerate(lines[dict_start_line:], start=dict_start_line):
    #     if '}' in line:
    #         dict_end_line = i
    #         break
    #     else:
    #         task_maps.add(lines[i])

    # # Insert the new dictionary entry before the end
    # task_insert = f'    "{task}{suffix}": {task}{suffix},\n'
    # if task_insert not in task_maps:
    #     lines.insert(dict_end_line, task_insert)

    # # Write the new contents back to the file
    # with open(task_map_file, 'w') as f:
    #     f.writelines(lines)