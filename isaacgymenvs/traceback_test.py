import os 
import subprocess
import time 

from train_rl_gpt import launch_rlg_hydra

ROOT_DIR = os.getcwd()

task = 'ShadowHandOver'
suffix= 'GPT'

processes = []
for i in range(3):
    process = subprocess.Popen(['python', f'{ROOT_DIR}/train_rl_gpt.py',  
                                            f'task={task}{suffix}', 'wandb_activate=True',
                                            'wandb_entity=jma2020', 'wandb_project=issac_gpt',
                                            'max_iterations=50',
                                            'headless=False', 'capture_video=False', 'force_render=False'],
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(2)
    processes.append(process)

for i, process in enumerate(processes):
    print(i)
    process.communicate()

# Execute the python file with flags
# process = subprocess.Popen(['python', '/home/exx/Projects/IsaacGymEnvs/isaacgymenvs/train.py',  
#                             f'task={task}', 'wandb_activate=True',
#                             'wandb_entity=jma2020', 'wandb_project=issac_gpt',
#                             'headless=True', 'test=True',
#                             'checkpoint=/data2/jasonyma/isaac_gpt/gpt_whole/2023-06-21_19-41-27/runs/BallBalanceGPT_21-19-46-40/nn/BallBalanceGPT.pth', 
#                             ],
#                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Continuous output
# for line in iter(process.stdout.readline, b''):
#     print(line.decode('utf-8'), end='')
# from ipdb import set_trace; set_trace()

# stdout_data, _ = process.communicate()
# stdout_str = stdout_data.decode('utf-8')
# print(stdout_str)

# def filter_traceback(s):
#     lines = s.split('\n')
#     for i, line in enumerate(lines):
#         if line.startswith('Traceback'):
#             return '\n'.join(lines[i:])
#     return ''  # Return an empty string if no Traceback is found

# print(filter_traceback(stdout_str))