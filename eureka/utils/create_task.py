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