from transformers import LlamaTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer
from extract_task_code import *
import csv
import os

llama_tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
model_id = "mistralai/Mixtral-8x7B-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Make list of llms we plan to use
llms = []
llms.append({'name': 'llama', 'tokenizer': LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")})
llms.append({'name': 'mistral', 'tokenizer': AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")})

EUREKA_ROOT_DIR = os.getcwd()
env_dir = f"{EUREKA_ROOT_DIR}/envs"

# Hyperparameters
number_of_iterations = 5
batch_size = 16

# Loading all text prompts and their length
prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
initial_system_str = file_to_string(f'{prompt_dir}/initial_system.txt')
code_output_tip_str = file_to_string(f'{prompt_dir}/code_output_tip.txt')
code_feedback_str = file_to_string(f'{prompt_dir}/code_feedback.txt')
initial_user_str = file_to_string(f'{prompt_dir}/initial_user.txt')
reward_signature_str = file_to_string(f'{prompt_dir}/reward_signature.txt')
policy_feedback_str = file_to_string(f'{prompt_dir}/policy_feedback.txt')
execution_error_feedback_str = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')
first_output_str = file_to_string(f'{prompt_dir}/eureka_first_output_example.txt')
feedback_str = file_to_string(f'{prompt_dir}/feedback_example.txt')
second_output_str = file_to_string(f'{prompt_dir}/eureka_second_output_example.txt')

# Iterate over list of llms
for llm in llms:
    name = llm['name']
    tokenizer = llm['tokenizer']
    print(f'Calculating for {name}')

    # Calculating number of tokens for every constant part
    initial_system = len(tokenizer.encode(initial_system_str))
    code_output_tip = len(tokenizer.encode(code_output_tip_str))
    code_feedback = len(tokenizer.encode(code_feedback_str))
    initial_user = len(tokenizer.encode(initial_user_str))
    reward_signature = len(tokenizer.encode(reward_signature_str))
    policy_feedback = len(tokenizer.encode(policy_feedback_str))
    execution_error_feedback = len(tokenizer.encode(execution_error_feedback_str))
    first_output = len(tokenizer.encode(first_output_str))
    feedback = len(tokenizer.encode(feedback_str))
    second_output = len(tokenizer.encode(second_output_str))

    # Create .csv file for writing
    file_name = f"{EUREKA_ROOT_DIR}/outputs/eureka/number_of_tokens/number_of_tokens_{name}.csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w', newline='') as file:

        writer = csv.writer(file)
        # Lists for rows
        shorten_code_envs = []
        full_code_envs = []

        for dirpath, dirnames, filenames in os.walk(env_dir):
            for subdir in dirnames:
                dir = f"{env_dir}/{subdir}"
                for dirpath_sub, dirnames_sub, filenames_sub in os.walk(dir):
                    for file in filenames_sub:
                        filename = f"{dir}/{file}"
                        file_string = file_to_string(filename)
                        env_code = len(tokenizer.encode(file_string))
                        first = initial_system + code_output_tip + initial_user + env_code
                        next = first + feedback + first_output
                        total = (first + next * (number_of_iterations - 1) * batch_size)
                        if filename.endswith("_obs.py"):
                            shorten_code_envs.append([filename, env_code, first, next, total])
                        else:
                            full_code_envs.append([filename, env_code, first, next, total])


        # Write data to fileb
        field = ["shorten env code"]
        writer.writerow(field)
        field = ["name", "env code", "first prompt", "next prompt", "total (5 iter, 16 batch size)"]
        writer.writerow(field)
        writer.writerows(shorten_code_envs)                                                   
        field = [""]
        writer.writerow(field)
        field = ["full env code"]
        writer.writerow(field)
        field = ["name", "env code", "first prompt", "next prompt", "total (5 iter, 16 batch size)"]
        writer.writerow(field)
        writer.writerows(full_code_envs)