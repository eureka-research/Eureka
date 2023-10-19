import subprocess
import os
import json
import logging

from utils.extract_task_code import file_to_string

def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    # Find GPU with most free memory
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])

    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "fps step:" in rl_log or "Traceback" in rl_log:
            if log_status and "fps step:" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and "Traceback" in rl_log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break

if __name__ == "__main__":
    print(get_freest_gpu())