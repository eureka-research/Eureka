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
        task_code_string  = file_to_string(task_file)
        task_obs_code_string  = file_to_string(task_obs_file)
    except:
        task_file = f'{ROOT_DIR}/tasks/bidex_gpt_base/{cfg.env.env_name.lower()}.py'
        task_obs_file = f'{ROOT_DIR}/tasks/bidex_gpt_base/{cfg.env.env_name.lower()}_obs.py'
        shutil.copy(task_obs_file, f"env_init_obs.py")
        task_code_string  = file_to_string(task_file)
        task_obs_code_string  = file_to_string(task_obs_file)

    output_file = f"{ROOT_DIR}/tasks/{cfg.env.env_name.lower()}{suffix.lower()}.py"

    # Loading all text prompts
    if cfg.paraphrase == -1:
        print("Using default prompts!")
        with open(f'{ROOT_DIR}/gpt_utils/prompts/initial_system.txt', 'r') as file:
            initial_system = file.read()
        with open(f'{ROOT_DIR}/gpt_utils/prompts/code_output_tip.txt', 'r') as file:
            code_output_tip = file.read()
        with open(f'{ROOT_DIR}/gpt_utils/prompts/code_feedback.txt', 'r') as file:
            code_feedback = file.read()
    else:
        with open(f'{ROOT_DIR}/gpt_utils/prompts/initial_system-{cfg.paraphrase}.txt', 'r') as file:
            initial_system = file.read()
        with open(f'{ROOT_DIR}/gpt_utils/prompts/code_output_tip-{cfg.paraphrase}.txt', 'r') as file:
            code_output_tip = file.read()
        with open(f'{ROOT_DIR}/gpt_utils/prompts/code_feedback-{cfg.paraphrase}.txt', 'r') as file:
            code_feedback = file.read()

    with open(f'{ROOT_DIR}/gpt_utils/prompts/initial_user.txt', 'r') as file:
        initial_user = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/reward_signature.txt', 'r') as file:
        reward_signature = file.read() 
    with open(f'{ROOT_DIR}/gpt_utils/prompts/policy_feedback.txt', 'r') as file:
        policy_feedback = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/execution_error_feedback.txt', 'r') as file:
        execution_error_feedback = file.read()


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

    DUMMY_FAILURE = -10000.
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_reward_code_path = None 
    
    # RewardGPT generation loop
    for iter in range(cfg.iteration):
        # Get GPT Response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0

        if "gpt3.5" in cfg.model:
            chunk_size = cfg.sample
        else:
            chunk_size = 4

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
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
        
        exec_success = False 
        valid_ids = [] # Keep track of valid code generation
        code_paths = []
        code_runs = [] 
        rl_runs = []

        checkpoint = cfg.checkpoint

        # Run the generated reward code
        for response_id in range(cfg.sample):
            response_cur = responses[response_id]["message"]["content"]
            # logging.info(f"Iteration {iter}: Processing Code Run {response_id}")
            logging.info(f"Iteration {iter}: Content {response_id}:\n " + response_cur + "\n")
            # Regex pattern to extract python code enclosed in GPT response
            pattern = r'```python(.*?)```'
            code_string = re.search(pattern, response_cur, re.DOTALL)
            if not code_string:
                pattern = r'```(.*?)```'
                code_string = re.search(pattern, response_cur, re.DOTALL)
            if not code_string:
                pattern = r'"""(.*?)"""'
                code_string = re.search(pattern, response_cur, re.DOTALL)
            if not code_string:
                pattern = r'""(.*?)""'
                code_string = re.search(pattern, response_cur, re.DOTALL)
            if not code_string:
                pattern = r'"(.*?)"'
                code_string = re.search(pattern, response_cur, re.DOTALL)

            if not code_string:
                code_string = response_cur
            else:
                code_string = code_string.group(1).strip()

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
                continue

            code_runs.append(code_string)
            reward_signature = f"        self.rew_buf[:], self.rew_dict = {gpt_reward_signature} \n        self.extras['gpt_reward'] = self.rew_buf.mean() \n        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()"
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            else:
                raise NotImplementedError

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

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            if cfg.text_only:
                continue
            # Find the freest GPU
            freest_gpu = get_freest_gpu()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{ROOT_DIR}/train_rl_gpt.py',  
                                            'hydra/output=subprocess',
                                            # f'env_path={env_real_path}',
                                            f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                            'wandb_entity=jma2020', 'wandb_project=isaac_gpt',
                                            f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                            f'max_iterations={cfg.max_iterations}', 
                                            f'checkpoint={checkpoint}'],
                                            stdout=f, stderr=f)
                
            # Hack to ensure that the RL training has started
            while True:
                with open(rl_filepath, 'r') as file:
                    rl_log = file.read()
                    if "fps step:" in rl_log and "Traceback" not in rl_log:
                        exec_success = True
                        valid_ids.append(response_id)
                        logging.info(f"Iteration {iter}: Code Run {response_id} succefully training!")
                        break
                    if "Traceback" in rl_log:
                        logging.info(f"Iteration {iter}: Code Run {response_id} execution error!")
                        break 
                    time.sleep(1)
            rl_runs.append(process)
        
        if cfg.text_only:
            return
        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue
        
        # Track the valid policies 
        valid_checkpoints = []
        
        # Evaluate the valid policies 
        # bash_file = f"{ROOT_DIR}/eval_script/{cfg.env.env_name.lower()}{suffix.lower()}_rlhf.sh"
        bash_file = f"{ROOT_DIR}/eval_script/rlhf_policy_vis.sh"
        with open(bash_file, "w") as bash:
            for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
                if response_id not in valid_ids:
                    continue 
                rl_run.communicate()
                rl_filepath = f"env_iter{iter}_response{response_id}.txt"
                code_paths.append(f"env_iter{iter}_response{response_id}.py")
                try:
                    with open(rl_filepath, 'r') as f:
                        stdout_str = f.read() 
                except:
                    continue 
            
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Network Directory:'):
                        break 
                network_dir = line.split(':')[-1].strip() 
                
                trained_policy_path = find_files_with_substring(network_dir, f"last_{task}{suffix}")
                if len(trained_policy_path) == 0:
                    checkpoint = ''
                else:
                    assert len(trained_policy_path) == 1, "More than one policy found!"
                    checkpoint = trained_policy_path[0]
                folder_name = f"{task}_iter{iter}_response{response_id}_vis"
                bash_line = (f'python train_rl_gpt.py capture_video=True capture_video_len=1000 headless=False force_render=False test=True hydra/output=eval custom_string={folder_name} num_envs=64 hydra.job.name="gpt_rlhf_policy" '
                            f'task={task} '
                            f'checkpoint="{checkpoint}" &\n')
                bash.write(bash_line)

        # Query human feedback
        print("Please open up terminal and type 'isaac; ./eval_script/rlhf_policy_vis.sh' to visualize the policies!")
        while True:
            print("Valid Policy IDs:", valid_ids)
            try:
                best_sample_idx = int(input("Please select the policy you like the most:"))
                if best_sample_idx in valid_ids:
                    confirm = input("Confirm your selection? (y/n)")
                    if confirm == "y":
                        break
                else:
                    print("Invalid policy ID, please try again")
            except:
                print("Error in input, please try again")
        while True:
            content = input("Please provide textual feedback on the behavior that you selected: \n")  
            confirm = input("Confirm your feedback? (y/n)")
            if confirm == "y":
                break

        best_content = content 
        max_reward_code_path = code_paths[best_sample_idx]
        shutil.copy(max_reward_code_path, f"env_iter{iter}_best.py")

        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
        messages += [{"role": "user", "content": best_content}]

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    
    # Evaluate the best reward code many times
    # if max_reward_code_path is None: 
    #     max_reward_code_path = "env_iter0_response0.py"
    # logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    # shutil.copy(max_reward_code_path, output_file)
    
    # eval_runs = []
    # for i in range(cfg.num_eval):
    #     # Find the freest GPU
    #     freest_gpu = get_freest_gpu()
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
        
    #     # Execute the python file with flags
    #     rl_filepath = f"reward_code_eval{i}.txt"
    #     with open(rl_filepath, 'w') as f:
    #         process = subprocess.Popen(['python', '-u', f'{ROOT_DIR}/train_rl_gpt.py',  
    #                                     'hydra/output=subprocess',
    #                                     # f'env_path={env_real_path}',
    #                                     f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
    #                                     'wandb_entity=jma2020', 'wandb_project=isaac_gpt',
    #                                     f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
    #                                     f'max_iterations={cfg.max_iterations}'],
    #                                     stdout=f, stderr=f)
                                        
    #     # Hack to ensure that the RL training has started
    #     while True:
    #         with open(rl_filepath, 'r') as file:
    #             rl_log = file.read()
    #             if "fps step:" in rl_log:
    #                 break
    #             if "Traceback" in rl_log:
    #                 break
    #     eval_runs.append(process)

    # reward_code_final_successes = []
    # reward_code_correlations_final = []
    # for i, rl_run in enumerate(eval_runs):
    #     rl_run.communicate()
    #     rl_filepath = f"reward_code_eval{i}.txt"
    #     with open(rl_filepath, 'r') as f:
    #         stdout_str = f.read() 
    #     lines = stdout_str.split('\n')
    #     for i, line in enumerate(lines):
    #         if line.startswith('Tensorboard Directory:'):
    #             break 
    #     tensorboard_logdir = line.split(':')[-1].strip() 
    #     tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
    #     max_success = max(tensorboard_logs['consecutive_successes'])
    #     reward_code_final_successes.append(max_success)

    #     if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
    #         gt_reward = np.array(tensorboard_logs["gt_reward"])
    #         gpt_reward = np.array(tensorboard_logs["gpt_reward"])
    #         reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
    #         reward_code_correlations_final.append(reward_correlation)

    # logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    # logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    # np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()