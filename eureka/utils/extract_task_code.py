import re 

def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def extract_task_code(filename):
    in_code = False
    in_task = False
    task_string = ''
    reward_string = ''
    with open(filename, 'r') as file:
        for line in file:
            # Skip leading comments (lines starting with '#')
            if not in_code and line.lstrip().startswith('#'):
                continue
            
            # Start adding lines to code_string once we've found a non-commented line
            if not in_code and not line.lstrip().startswith('#'):
                in_code = True
                in_task = True
            # Stop adding lines once we've found a line with multiple '#'
            if in_code and line.count('#') > 1:
                in_task = False
                continue
                # reward_string += line
                # break
            
            # Add line to code_string
            if in_code and in_task:
                task_string += line
            else:
                reward_string += line    
    return task_string, reward_string

def extract_observation_code(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    function_code = ''
    function_started = False

    for line in lines:
        if re.match(r'def .+observations.+\(.*\):', line):
            function_started = True
            print(line)
        if function_started:
            function_code += line

            if line.strip() == '':
                function_started = False

    return function_code

def extract_observation_functions(filename, task='ant'):
    with open(filename, 'r') as f:
        lines = f.readlines()

    functions = []
    function_lines = []
    indent = 0

    for line in lines:
        stripped_line = line.lstrip()
        current_indent = len(line) - len(stripped_line)

        if current_indent < indent:  # if the indent decreases, we've left the function
            if function_lines:  # if there are lines saved, we save the function
                functions.append(''.join(function_lines))
                function_lines = []  # clear function lines

        if re.match(r'def .+ant_observations.+\(.*\):', stripped_line):
            indent = current_indent
            function_lines.append(line)  # save function line
        elif function_lines:  # if we're in the function, save lines
            function_lines.append(line)

    # handle case where the file ends but we're still in a function
    if function_lines:
        functions.append(''.join(function_lines))

    return '\n'.join(functions)

import ast

def get_function_signature(code_string):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = function_def.name + '(self.' + ', self.'.join(arg.arg for arg in function_def.args.args) + ')'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst

