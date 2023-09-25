import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches

def load_tensorboard_logs(path):
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    # Show all tags in the log file
    # print(event_acc.Tags())

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
    
    return data

import importlib.util

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function

# Example usage
if __name__ == '__main__':
    # directory_path = '/data2/jasonyma/isaac_gpt/gpt_whole/2023-06-21_19-41-27'
    # substring = 'BallBalanceGPT.pth'
    # matched_files = find_files_with_substring(directory_path, substring)
    # print(matched_files)
    file_path = "/data2/jasonyma/isaac_gpt/gpt_bidex/2023-07-19_15-53-09/env_iter0_response0.py"
    import_class_from_file(file_path, "ShadowHandOverGPT")