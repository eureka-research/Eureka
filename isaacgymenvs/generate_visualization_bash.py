import os

tasks = set() 

def generate_bash_script(directory):
    run_folder = directory.split("/")[-1]
    bash_file = f"eval_script/{run_folder}.sh"

    # Open the bash script for writing
    with open(bash_file, "w") as f:
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".pth"):
                    if "last" in file:
                        continue
                    full_path = os.path.join(root, file)
                    hydra_job_name = full_path.split("/")[-2]
                    # Extract task name. This takes the filename, strips ".pth", 
                    # then splits on "GPT" and takes the first part. Finally, it strips any number and underscore from the end.
                    task_name = file.rsplit("GPT", 1)[0].rsplit("_", 1)[0].rsplit("-")[0]
                    file_name = file.rsplit("/")[0][:-4]
                    if task_name not in tasks:
                        tasks.add(task_name)
                    else:
                        continue 
                    bash_line = (f'python train_rl_gpt.py capture_video=True headless=False force_render=False test=True hydra/output=eval custom_string={file_name} hydra.job.name={hydra_job_name}_final '
                                f'task={task_name} '
                                f'checkpoint="{full_path}"\n')
                    f.write(bash_line)

    print(f"Bash script saved in {bash_file}")

# Specify the directory you want to scan
# directory_to_scan = "/data2/jasonyma/isaac_gpt/gpt_policy/reward-gpt4-final-bidex-markov-gtfeedback"
directory_to_scan = "/data2/jasonyma/isaac_gpt/gpt_policy/ngc_bidex_human_baseline"

# directory_to_scan = "/data2/jasonyma/isaac_gpt/gpt_policy/reward-gpt4-final-isaac-markov-fixed"
# directory_to_scan = "/data2/jasonyma/isaac_gpt/gpt_policy/reward-gpt4-final-isaac-markov-gtfeedback"
# directory_to_scan = "/data2/jasonyma/isaac_gpt/gpt_policy/ngc_isaac_human_baseline"
generate_bash_script(directory_to_scan)