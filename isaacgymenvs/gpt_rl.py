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

    # Replace the task name once upfront when saving new GPT-generated reward code
    if 'amp' not in cfg.env.env_name:    
        # Create Task YAML files
        create_task(ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    output_file = f"{ROOT_DIR}/tasks/{cfg.env.env_name.lower()}{suffix.lower()}.py"

    assert cfg.original_path is not None, "Must specify original Eureka run path"
    # Retrieve the best reward code from the original Eureka run
    original_path = cfg.original_path
    logging.info("Original path: " + original_path)
    summary = np.load(original_path + "/" + "summary.npz", allow_pickle=True)
    max_successes = summary['max_successes']
    best_code_paths = summary['best_code_paths']
    max_successes_reward_correlation = summary['max_successes_reward_correlation']
    max_success_overall = max(max_successes)
    max_success_idx = np.argmax(max_successes)
    max_success_reward_correlation_overall = max_successes_reward_correlation[max_success_idx]
    max_reward_code_path = best_code_paths[max_success_idx] 
    max_reward_code_path = cfg.original_path + "/" + max_reward_code_path
    
    logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")    
    shutil.copy(max_reward_code_path, output_file)

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
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                                        ],
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