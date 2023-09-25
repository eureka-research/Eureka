import ast

def get_last_line_of_docstring(file_path, function_name, search_string):
    print(file_path)
    with open(file_path, 'r') as file:
        file_content = file.read()
        module = ast.parse(file_content)
        
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef) and sub_node.name == function_name:
                    docstring = ast.get_docstring(sub_node)
                    if docstring is None:
                        print(f"The function '{function_name}' does not have a docstring.")
                    else:
                        docstring_lines = docstring.split('\n')
                        print(f"{docstring_lines[-1]}")
                        break
                    # return
                    # for line in ast.unparse(sub_node).split('\n'):
                    #     if search_string in line:
                    #         print(f"Line containing '{search_string}':\n{line}")
                            
            else:
                print(f"The function '{function_name}' was not found in the class.")
    for line in file_content.split('\n'):
        if search_string in line:
            print(f"{line}")
            break

tasks = ["shadow_hand_block_stack", "shadow_hand_bottle_cap", "shadow_hand_catch_abreast",
         "shadow_hand_catch_over2underarm", "shadow_hand_catch_underarm", 
         "shadow_hand_door_close_inward", "shadow_hand_door_close_outward",
         "shadow_hand_door_open_inward", "shadow_hand_door_open_outward",
         "shadow_hand_grasp_and_place", "shadow_hand_kettle", 
         "shadow_hand_lift_underarm", "shadow_hand_over",
         "shadow_hand_pen", "shadow_hand_push_block",
         "shadow_hand_push_block", "shadow_hand_re_orientation",
         "shadow_hand_scissors",
         "shadow_hand_swing_cup", "shadow_hand_switch",
         "shadow_hand_two_catch_underarm"]

# tasks = ["shadow_hand_over"]

for task in tasks:
    get_last_line_of_docstring(f"tasks/{task}.py", "compute_full_state", 'full_state":')
    print()