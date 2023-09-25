import hydra
import numpy as np 
import json
import logging 
import math 
import matplotlib.pyplot as plt
import os
import openai
import re
import subprocess
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import shutil
import time 
from tqdm import tqdm 
from gpt_utils.misc import * 
from gpt_utils.file_utils import find_files_with_substring, load_tensorboard_logs
from gpt_utils.create_task import create_task
from gpt_utils.extract_task_code import *
from gpt_utils.prompts.examples import example_code

ROOT_DIR = os.getcwd()

@hydra.main(config_path="cfg_gpt", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    openai.organization = "minedojo"
    # openai.organization = "university-of-pennsylvania-170"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.Model.list()
    model = cfg.model
    logging.info("Using LLM: " + model)

    task = cfg.env.task
    task_description = cfg.env.description
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)
    suffix = cfg.suffix

    try:
        task_file = f'{ROOT_DIR}/tasks/isaac_gpt_base/{cfg.env.env_name.lower()}.py'
        task_obs_file = f'{ROOT_DIR}/tasks/isaac_gpt_base/{cfg.env.env_name.lower()}_obs.py'
        shutil.copy(task_obs_file, f"env_init_obs.py")
        task_code_string  = python_file_to_string(task_file)
        task_obs_code_string  = python_file_to_string(task_obs_file)
    except:
        task_file = f'{ROOT_DIR}/tasks/bidex_gpt_base/{cfg.env.env_name.lower()}.py'
        task_obs_file = f'{ROOT_DIR}/tasks/bidex_gpt_base/{cfg.env.env_name.lower()}_obs.py'
        shutil.copy(task_obs_file, f"env_init_obs.py")
        task_code_string  = python_file_to_string(task_file)
        task_obs_code_string  = python_file_to_string(task_obs_file)

    output_file = f"{ROOT_DIR}/tasks/{cfg.env.env_name.lower()}{suffix.lower()}.py"

    # Loading all text prompts
    with open(f'{ROOT_DIR}/gpt_utils/prompts/initial_system.txt', 'r') as file:
        initial_system = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/initial_user_curriculum.txt', 'r') as file:
        initial_user = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/reward_signature.txt', 'r') as file:
        reward_signature = file.read() 
    with open(f'{ROOT_DIR}/gpt_utils/prompts/policy_feedback.txt', 'r') as file:
        policy_feedback = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/execution_error_feedback.txt', 'r') as file:
        execution_error_feedback = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/code_output_tip.txt', 'r') as file:
        code_output_tip = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/code_feedback.txt', 'r') as file:
        code_feedback = file.read()

    if cfg.load_message != "":
        with open(cfg.load_message) as message_history:
            messages = json.load(message_history)
            if messages[-1]['role'] != 'user':
                content = input("Please provide a user input to the loadded message history: \n") 
                logging.info(f"Iteration {iter}: User Content:\n" + content + "\n")
                messages += [{"role": "user", "content": content}]
            # shorten loaded messages (keep the initial system,user, assistant triplets, and the last three messages)
            messages = messages[:3] + messages[-3:]
    else:
        # Start from scratch           
        messages = [{"role":"system", "content": initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip},           
                    {"role":"user", "content": initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)}]
        
        logging.info("Initial System:\n " + messages[0]["content"] + "\n")
        logging.info("Initial User:\n " + messages[1]["content"] + "\n")

    # Replace the task name once upfront when saving new GPT-generated reward code
    if 'amp' not in cfg.env.env_name:
        task_code_string = task_code_string.replace(task, task+suffix)
    
        # Create Task YAML files
        create_task(ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    max_successes = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = -1.
    reward_code_path = None 
    
    # RewardGPT generation loop
    assert cfg.iteration == 1, "Curriculum mode currently supports only 1 iteration!"
    for iter in range(cfg.iteration):

        # Get GPT Response
        total_token = 0
        total_completion_token = 0
        responses = []
        if "gpt3.5" in cfg.model:
            chunk_size = cfg.sample
        else:
            chunk_size = 4

        # total_samples = 0
        # while True:
        #     if total_samples >= cfg.sample:
        #         break
        #     for attempt in range(1000):
        #         try:
        #             response_cur = openai.ChatCompletion.create(
        #                 model=model,
        #                 messages=messages,
        #                 temperature=cfg.temperature,
        #                 n=chunk_size
        #             )
        #             total_samples += chunk_size
        #             break
        #         except Exception as e:
        #             if attempt >= 200:
        #                 chunk_size = int (chunk_size / 2)
        #             logging.info(f"Attempt {attempt+1} failed with error: {e}")
        #             time.sleep(1)  # sleep for 1 second before retrying
        #     if response_cur is None:
        #         logging.info("Code terminated due to too many failed attempts!")
        #         exit()
        #     responses.extend(response_cur["choices"])
        #     prompt_tokens = response_cur["usage"]["prompt_tokens"]
        #     total_completion_token += response_cur["usage"]["completion_tokens"]
        #     total_token += response_cur["usage"]["total_tokens"]

        chunks = math.ceil(cfg.sample / chunk_size)
        for t in tqdm(range(chunks)):
            num_samples = min(cfg.sample-t*chunk_size, chunk_size)
            for attempt in range(1000):
                try:
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=num_samples
                    )
                    break
                except Exception as e:
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)  # sleep for 1 second before retrying
            
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur["choices"])
            prompt_tokens = response_cur["usage"]["prompt_tokens"]
            total_completion_token += response_cur["usage"]["completion_tokens"]
            total_token += response_cur["usage"]["total_tokens"]

        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [[] for _ in range(len(responses))] 
        rl_runs = [[] for _ in range(len(responses))]
        for response_id in range(len(responses)):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")
            with open(f"dialogue{response_id}.txt", "a") as file:
                file.write(f"GPT Iteration {iter} Output:\n")
                file.write(f"============================================================================================\n")
                file.write(response_cur + "\n")
                file.write(f"============================================================================================\n")

            # Allow chat mode; no RL feedback
            if cfg.text_only:
                assert cfg.sample == 1, "Text only mode only supports sample=1"
                content = input("Please provide feedback on the reward function: \n") 
                messages += [{"role": "assistant", "content": response_cur}]
                logging.info(f"Iteration {iter}: User Content:\n" + content + "\n")
                messages += [{"role": "user", "content": content}]

                # Save dictionary as JSON file
                with open('messages.json', 'w') as file:
                    json.dump(messages, file, indent=4)
                continue 
            # Regex pattern to extract python code enclosed in GPT response
            pattern = r'```python(.*?)```'
            curriculum_string = re.findall(pattern, response_cur, re.DOTALL)
            if not curriculum_string:
                pattern = r'```(.*?)```'
                curriculum_string = re.findall(pattern, response_cur, re.DOTALL)
            if not curriculum_string:
                pattern = r'"""(.*?)"""'
                curriculum_string = re.findall(pattern, response_cur, re.DOTALL)
            if not curriculum_string:
                pattern = r'""(.*?)""'
                curriculum_string = re.findall(pattern, response_cur, re.DOTALL)
            if not curriculum_string:
                pattern = r'"(.*?)"'
                curriculum_string = re.findall(pattern, response_cur, re.DOTALL)

            if not curriculum_string:
                curriculum_string = response_cur

            # from ipdb import set_trace
            # set_trace()
            for step_id, code_string in enumerate(curriculum_string):
                code_string = code_string.strip()
                # Remove unnecessary imports
                lines = code_string.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        code_string = "\n".join(lines[i:])
                    
                # Add the GPT Reward Signature to the environment code
                try:
                    gpt_reward_signature, input_lst = get_function_signature(code_string)
                    
                    # TODO (Jason): what does the following line do?
                    for input_cur in input_lst:
                        if input_cur not in code_string:
                            continue 
                except Exception as e:
                    # TODO: is there a easy way to fix this?
                    logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                    # rl_filepath = f"env_iter{iter}_response{response_id}.txt"
                    # with open(rl_filepath, 'w') as f:
                    #     f.writelines(f"Error: {e} \n Code Run cannot parse function signature!")
                    continue

                code_runs[response_id].append(code_string)
                reward_signature = f"        self.rew_buf[:], self.rew_dict = {gpt_reward_signature} \n        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()"
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)

                # Save the new environment code when the output contains valid code string!
                with open(output_file, 'w') as file:
                    file.writelines(task_code_string_iter + '\n')
                    file.writelines("from typing import Tuple, Dict" + '\n')
                    file.writelines("import math" + '\n')
                    file.writelines("import torch" + '\n')
                    file.writelines("from torch import Tensor" + '\n')
                    if "@torch.jit.script" not in code_string:
                        code_string = "@torch.jit.script\n" + code_string
                    file.writelines(code_string + '\n')

                with open(f"env_iter{iter}_response{response_id}_step{step_id}_rewardonly.py", 'w') as file:
                    file.writelines(code_string + '\n')

                # Copy the generated environment code to hydra output directory for bookkeeping
                shutil.copy(output_file, f"env_iter{iter}_response{response_id}_step{step_id}.py")
                
                # Find the freest GPU
                freest_gpu = get_freest_gpu()
                os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
                
                # Execute the python file with flags
                # TODO: Add current policy
                rl_filepath = f"env_iter{iter}_response{response_id}_step{step_id}.txt"
                if step_id == 0:
                    checkpoint = cfg.checkpoint

                with open(rl_filepath, 'w') as f:
                    process = subprocess.Popen(['python', '-u', f'{ROOT_DIR}/train_rl_gpt.py',  
                                                'hydra/output=subprocess',
                                                # f'env_path={env_real_path}',
                                                f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                                'wandb_entity=jma2020', 'wandb_project=issac_gpt',
                                                f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                                # f'max_iterations={cfg.max_iterations}',
                                                f'checkpoint={checkpoint}'],
                                                stdout=f, stderr=f)

                while True:
                    with open(rl_filepath, 'r') as file:
                        rl_log = file.read()
                        if "fps step:" in rl_log:
                            logging.info(f"Iteration {iter}: Code Run {response_id}, Step {step_id} succefully training!")
                            break
                        if "Traceback" in rl_log:
                            logging.info(f"Iteration {iter}: Code Run {response_id}, Step {step_id} execution error!")
                            break 

                process.communicate()
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
                content = ''
                traceback_msg = filter_traceback(stdout_str)
                if traceback_msg != '':
                    content += execution_error_feedback.format(traceback_msg=traceback_msg)
                    # break 
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Network Directory:'):
                        break 
                network_dir = line.split(':')[-1].strip() 
                
                # try:
                #     trained_policy_path = find_files_with_substring(network_dir, f"{task}{suffix}.pth") 
                # except:
                trained_policy_path = find_files_with_substring(network_dir, f"last_{task}{suffix}")
                if len(trained_policy_path) == 0:
                    checkpoint = ''
                else:
                    assert len(trained_policy_path) == 1, "More than one policy found!"
                    checkpoint = trained_policy_path[0]
                print(checkpoint)
                # rl_runs[response_id].append(process)
    
    
if __name__ == "__main__":
    main()