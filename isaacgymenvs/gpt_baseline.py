# Implementation of Google's "Language to Rewards" as a baseline

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
from gpt_utils.code_gen_utils import inject_code, indent_code
from gpt_utils.create_task import create_task
from gpt_utils.extract_task_code import *
from gpt_utils.prompts.examples import example_code

ROOT_DIR = os.getcwd()

def query_gpt(model, temperature, messages, samples, chunk_size=1, attempts=1000):
    responses = []
    total_samples = 0
    total_completion_token = 0
    total_token = 0
    while True:
        if total_samples >= samples:
            break
        for attempt in range(attempts):
            try:
                response_cur = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
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

    return (
        responses,
        prompt_tokens,
        total_completion_token,
        total_token,
    )

@hydra.main(config_path="cfg_gpt", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    openai.organization = "minedojo"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.Model.list()
    model = cfg.model
    logging.info("Using LLM: " + model)

    task = cfg.env.task
    task_description = cfg.env.description
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)
    suffix = cfg.suffix

    par_dir = "baseline_gpt_base"
    env_filename = cfg.env.env_name.lower()
    env_objects_file = f"{ROOT_DIR}/tasks/{par_dir}/bidex_env_fields.json"
    task_file = f'{ROOT_DIR}/tasks/{par_dir}/{env_filename}.py'
    task_descriptor_prompt_file = f'{ROOT_DIR}/tasks/{par_dir}/bidex_descriptor_prompt.md'
    task_coder_prompt_file = f'{ROOT_DIR}/tasks/{par_dir}/bidex_coder_prompt.md'
    coder_reward_api_file = f"{ROOT_DIR}/tasks/{par_dir}/bidex_reward_api.py"
    output_file = f"{ROOT_DIR}/tasks/{env_filename}{suffix.lower()}.py"

    task_code_string  = file_to_string(task_file)
    task_descriptor_prompt_string = file_to_string(task_descriptor_prompt_file)
    task_coder_prompt_string = file_to_string(task_coder_prompt_file)
    coder_reward_api_string = file_to_string(coder_reward_api_file)
    task_code_string = inject_code(task_code_string, coder_reward_api_string, "##### Insert Bidex reward API here #####")
    with open(env_objects_file) as f:
        data = json.load(f)
        env_fields = data[env_filename]
    env_hands = [x for x in env_fields if "thumb" in x or "finger" in x or "palm" in x]
    env_objects = [x for x in env_fields if x not in env_hands]
    env_rot_fields = [x for x in env_fields if x in ["object1", "object2", "pot", "cup", "object1_goal", "object2_goal", "pot_goal", "cup_goal"]]

    hand_descriptions = []
    for x in env_hands:
        side, name = x.split("_")
        hand_descriptions.append(f"[optional] to perform this task, the {side} manipulator's {name} should move close to {{CHOICE: <INSERT OBJECTS HERE>}}.")
    task_descriptor_prompt_string = task_descriptor_prompt_string.replace("<INSERT OPTIONAL HAND DESCRIPTIONS HERE>", "\n".join(hand_descriptions))
    task_descriptor_prompt_string = task_descriptor_prompt_string.replace("<INSERT OBJECTS HERE>", ", ".join(env_objects))
    task_coder_prompt_string = task_coder_prompt_string.replace("<INSERT FIELDS HERE>", ", ".join(env_fields))
    task_coder_prompt_string = task_coder_prompt_string.replace("<INSERT ORIENTATION FIELDS HERE>", ", ".join(env_rot_fields))

    # Loading all text prompts
    with open(f'{ROOT_DIR}/gpt_utils/prompts/initial_system.txt', 'r') as file:
        initial_system = file.read()
    with open(f'{ROOT_DIR}/gpt_utils/prompts/initial_user.txt', 'r') as file:
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
        # messages = [{"role":"system", "content": initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip},           
        #             {"role":"user", "content": f"You are to solve the following task: {task_description}"}]
        messages = [{"role":"user", "content": f"You are to solve the following task: {task_description}"}]
        
        # logging.info("Initial System:\n " + messages[0]["content"] + "\n")
        # logging.info("Initial User:\n " + messages[1]["content"] + "\n")

    # Replace the task name once upfront when saving new GPT-generated reward code
    if 'amp' not in cfg.env.env_name:
        task_code_string = task_code_string.replace(task, task+suffix)
    
        # Create Task YAML files
        create_task(ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    # for response_id in range(cfg.sample):
    #     with open(f"dialogue{response_id}.txt", "a") as file:
    #         file.write(f"System Input:\n")
    #         file.write(f"============================================================================================\n")
    #         file.write(messages[0]["content"] + "\n")
    #         file.write(f"============================================================================================\n")

    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = -1.
    max_success_reward_correlation_overall = -1.
    reward_code_path = None 
    
    # RewardGPT generation loop
    for iter in range(cfg.iteration):

        # Since L2R uses a two-stage LLM generation process, in order to simulate multiple samples
        # we get multiple samples of the first stage and then one second-stage sample per first-stage sample
        messages_for_descriptor = messages + [{"role": "system", "content": task_descriptor_prompt_string}]
        cur_descriptor_responses, _, _, _ = query_gpt(
            model,
            cfg.temperature,
            messages_for_descriptor,
            samples=cfg.sample,
            chunk_size=cfg.sample if "gpt-3.5" in cfg.model else 4
        )
        cur_coder_responses = []
        for response in cur_descriptor_responses:
            messages_for_coder = [{"role": "system", "content": task_coder_prompt_string}, {"role": "user", "content": response["message"]["content"]}]
            cur_coder_response, _, _, _ = query_gpt(
                model,
                cfg.temperature,
                messages_for_coder,
                samples=1
            )
            cur_coder_responses.extend(cur_coder_response)

        # Messages consists of task description, system descriptor prompt, system coder prompt, and generated descriptor response
        messages = messages_for_descriptor + messages_for_coder
        responses = cur_coder_responses
        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n" + responses[0]["message"]["content"] + "\n")

        # Logging Token Information
        # logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(len(responses)):
            response_cur = responses[response_id]["message"]["content"]
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")
            # with open(f"dialogue{response_id}.txt", "a") as file:
            #     file.write(f"GPT Iteration {iter} Output:\n")
            #     file.write(f"============================================================================================\n")
            #     file.write(response_cur + "\n")
            #     file.write(f"============================================================================================\n")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
            ]
            for pattern in patterns:
                reward_code_string = re.search(pattern, response_cur, re.DOTALL)
                if reward_code_string is not None:
                    reward_code_string = reward_code_string.group(1).strip()
                    break
            reward_code_string = response_cur if not reward_code_string else reward_code_string
            for fn in ["set_min_l2_distance_reward", "set_max_l2_distance_reward", "set_obj_orientation_reward"]:
                reward_code_string = reward_code_string.replace(fn, "self." + fn)

            # Add the GPT Reward Signature to the environment code
            if not cfg.text_only:
                code_runs.append(reward_code_string)

                # Save the new environment code when the output contains valid code string!
                full_code_string = inject_code(task_code_string, reward_code_string, "##### Insert GPT reward here #####")
                with open(output_file, 'w') as file:
                    file.writelines(full_code_string)

                with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                    file.writelines(reward_code_string + "\n")

                # Copy the generated environment code to hydra output directory for bookkeeping
                shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

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
                exit()
                continue 
            
            # Find the freest GPU
            freest_gpu = get_freest_gpu()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{ROOT_DIR}/train_rl_gpt.py',  
                                            'hydra/output=subprocess',
                                            # f'env_path={env_real_path}',
                                            f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                            'wandb_entity=jma2020', 'wandb_project=isaac_gpt',
                                            f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                            f'max_iterations={cfg.max_iterations}'],
                                            stdout=f, stderr=f)
                
            # Hack to ensure that the RL training has started
            while True:
                with open(rl_filepath, 'r') as file:
                    rl_log = file.read()
                    if "fps step:" in rl_log:
                        logging.info(f"Iteration {iter}: Code Run {response_id} successfully training!")
                        break
                    if "Traceback" in rl_log:
                        logging.info(f"Iteration {iter}: Code Run {response_id} execution error!")
                        break 
            rl_runs.append(process)
        
        # Gather RL training results and provide feedback
        if cfg.self_critic:
            # In self-critic mode, we directly provide feedback to the reward code without using policy training to provide feedback
            assert cfg.sample == 1, "Self-critic mode only supports sample=1"
            # User command line input
            content = input("Please provide feedback on the reward function: \n") 
        else:
            contents = []
            successes = []
            reward_correlations = []
            code_paths = []
            
            exec_success = False 
            for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
                rl_run.communicate()
                rl_filepath = f"env_iter{iter}_response{response_id}.txt"
                code_paths.append(f"env_iter{iter}_response{response_id}.py")
                try:
                    with open(rl_filepath, 'r') as f:
                        stdout_str = f.read() 
                except: 
                    content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                    content += code_output_tip
                    contents.append(content) 
                    successes.append(-1.)
                    reward_correlations.append(-1.)
                    continue

                content = ''
                traceback_msg = filter_traceback(stdout_str)

                # If RL execution has no error, provide policy statistics feedback
                if traceback_msg == '':
                    exec_success = True
                    # logging.info(f"Iteration {iter}, Response {i}: GPT Reward Function is valid! RL training is successful!\n")
                    if not cfg.human:
                        lines = stdout_str.split('\n')
                        for i, line in enumerate(lines):
                            if line.startswith('Tensorboard Directory:'):
                                break 
                        tensorboard_logdir = line.split(':')[-1].strip() 
                        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                        epoch_freq = int(cfg.max_iterations // 10)

                        content += policy_feedback.format(epoch_freq=epoch_freq)
                        
                        # Compute Correlation between Human-Engineered and GPT Rewards
                        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                            gt_reward = np.array(tensorboard_logs["gt_reward"])
                            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                            reward_correlations.append(reward_correlation)

                        # Add reward components log to the feedback
                        for metric in tensorboard_logs:
                            if "/" not in metric:
                                metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                                metric_cur_max = max(tensorboard_logs[metric])
                                metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                                if "consecutive_successes" == metric:
                                    successes.append(metric_cur_max)
                                metric_cur_min = min(tensorboard_logs[metric])
                                if metric != "gt_reward":
                                    content += f"{metric}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                                else:
                                    # Provide ground-truth score when success rate not applicable
                                    if "consecutive_successes" not in tensorboard_logs:
                                        content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                        content += code_feedback  
                    else:
                        assert cfg.sample == 1, "in-the loop Human feedback mode only supports sample=1"
                        # User command line input
                        content = input("Please provide human textual feedback on the reward function: \n")  
                    logging.info(f"Iteration {iter}: Code Run {response_id}, Max Success: {successes[-1]:.2f}, Correlation: {reward_correlation}")
                # Otherwise, provide execution traceback error feedback              
                else:
                    successes.append(-1.)
                    reward_correlations.append(-1.)
                    content += execution_error_feedback.format(traceback_msg=traceback_msg)

                # logging.info(f"Iteration {iter+1}, User Response {run_id}: {content}\n")
                content += code_output_tip
                contents.append(content) 
            
            # Repeat the iteration if all code generation failed
            if not exec_success:
            # if not exec_success and cfg.sample != 1:
                execute_rates.append(0.)
                max_successes.append(-1.)
                logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
                continue

            # Select the best code sample based on the success rate
            best_sample_idx = np.argmax(np.array(successes))
            best_content = contents[best_sample_idx]
            max_success = successes[best_sample_idx]
            max_success_reward_correlation = reward_correlations[best_sample_idx]
            execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

            if max_success > max_success_overall:
                max_success_overall = max_success
                max_success_reward_correlation_overall = max_success_reward_correlation
                reward_code_path = code_paths[best_sample_idx]

            execute_rates.append(execute_rate)
            max_successes.append(max_success)
            max_successes_reward_correlation.append(max_success_reward_correlation)
            best_code_paths.append(reward_code_path)

            # Plot the success rate
            fig, axs = plt.subplots(2, figsize=(6, 6))
            fig.suptitle(f'{cfg.env.task}')

            x_axis = np.arange(len(max_successes))

            axs[0].plot(x_axis, np.array(max_successes))
            axs[0].set_title("Max Success")
            axs[0].set_xlabel("Iteration")

            axs[1].plot(x_axis, np.array(execute_rates))
            axs[1].set_title("Execute Rate")
            axs[1].set_xlabel("Iteration")

            fig.tight_layout(pad=3.0)
            plt.savefig('summary.png')
            np.savez('summary.npz', max_successes=max_successes, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)

            logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
            logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
            logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
            logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
            messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    if cfg.human:
        return 
    
    # Evaluate the best reward code many times
    if reward_code_path is None: 
        reward_code_path = "env_iter0_response0.py"
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {reward_code_path}")
    shutil.copy(reward_code_path, output_file)
    
    eval_runs = []
    for i in range(cfg.num_eval):
        # Find the freest GPU
        freest_gpu = get_freest_gpu()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{ROOT_DIR}/train_rl_gpt.py',  
                                        'hydra/output=subprocess',
                                        # f'env_path={env_real_path}',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        'wandb_entity=jma2020', 'wandb_project=isaac_gpt',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                        f'max_iterations={cfg.max_iterations}'],
                                        stdout=f, stderr=f)
            
        # Hack to ensure that the RL training has started
        while True:
            with open(rl_filepath, 'r') as file:
                rl_log = file.read()
                if "fps step:" in rl_log:
                    break
                if "Traceback" in rl_log:
                    break
        eval_runs.append(process)

    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'r') as f:
            stdout_str = f.read() 
        lines = stdout_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('Tensorboard Directory:'):
                break 
        tensorboard_logdir = line.split(':')[-1].strip() 
        tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
        max_success = max(tensorboard_logs['consecutive_successes'])
        reward_code_final_successes.append(max_success)

        if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
            gt_reward = np.array(tensorboard_logs["gt_reward"])
            gpt_reward = np.array(tensorboard_logs["gpt_reward"])
            reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
            reward_code_correlations_final.append(reward_correlation)

    logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)


if __name__ == "__main__":
    main()