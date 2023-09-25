import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm 
import re 
import pickle 

# Specify the parent directory that contains all your TensorBoard log files.
# parent_dir = "/Users/jasoma/nvidia-workspace/paper_code"


# Get all tensorboard files in the parent directory
# parent_dir = "/Users/jasoma/nvidia-workspace/ngc_results"
# file_paths = glob.glob(f"{parent_dir}/*/*/*/*/*/*/*/events.out.tfevents*", recursive=True)

file_paths = glob.glob(f"/home/user/workspace/isaac_gpt/ngc_bidex_human_baseline_sparse/*/runs/*/summaries/events.out.tfevents*", recursive=True)
# Dict to store all the metrics, keys are tasks
all_values = defaultdict(list)

# Process each file.
for path in tqdm(file_paths):
    match = re.search(r'runs/(\w+)-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', path)
    if match:
        task = match.group(1)
        print(path)
        print(task)
    else:
        print("Pattern not found in the given string.")
    try:
        # event_acc = EventAccumulator(path, size_guidance={'scalars': 500})
        event_acc = EventAccumulator(path)
        event_acc.Reload()  # Load the data.

        # Get the metric you're interested in.
        consecutive_successes = event_acc.Scalars('consecutive_successes')
    except:
        continue 
    steps = [event.step for event in consecutive_successes]
    values = [event.value for event in consecutive_successes]
    
    # Identify the task from the directory name
    if len(values) != 6000:
        continue
    # Add the values to the dict.
    print(task, len(values))
    all_values[task].append(values)

file_name = "/home/user/workspace/results/ppo_bidex_sparse.pkl"
with open(file_name, "wb") as outfile:
    pickle.dump(all_values, outfile)
exit()

# with open("ppo_bidex_humanreward_v2.pkl", "rb") as infile:
    # all_values = pickle.load(infile)

# Create a dictionary to hold the average consecutive successes for each task
# avg_consecutive_successes = {}

# # Process each task.
# for task, values in all_values.items():
#     # Compute the mean and standard deviation across all runs.
#     values = np.array(values)
#     steps = np.arange(values.shape[1])
#     print(task, values.shape)
#     mean_values = np.mean(values, axis=0)
#     std_values = np.std(values, axis=0)

#     # Plot the mean and shade the standard deviation.
#     plt.figure()
#     plt.plot(steps, mean_values, label='Average')
#     plt.fill_between(steps, mean_values - std_values, mean_values + std_values, color='gray', alpha=0.5, label='Std Dev')
#     plt.xlabel('Iterations')
#     plt.ylabel('Consecutive Successes')
#     plt.title(f'Average Consecutive Successes Over Time for {task}')
#     plt.legend()
#     plt.savefig(f'results/bidex_{task}_consecutive_successes.png')
    
#     # Store the average of mean values for the bar plot
#     avg_consecutive_successes[task] = np.mean(mean_values)

# # Create the bar plot for average consecutive successes across all tasks
# # Sort tasks by average consecutive successes
# sorted_tasks = sorted(avg_consecutive_successes.items(), key=itemgetter(1), reverse=True)
# tasks, avg_values = zip(*sorted_tasks)

# plt.figure()
# plt.barh(tasks, avg_values)
# plt.xlabel('Average Consecutive Successes')
# plt.ylabel('Tasks')
# plt.title('Average Consecutive Successes for All Tasks')
# plt.gca().invert_yaxis()  # Highest value at top
# plt.tight_layout()
# plt.savefig('results/bidex_avg_consecutive_successes_all_tasks.png')