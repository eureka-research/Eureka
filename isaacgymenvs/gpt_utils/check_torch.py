import os
import re

def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return the lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Added error handling here
    try:
        with open(file_name, 'r') as read_obj:
            for line in read_obj:
                line_number += 1
                if string_to_search in line:
                    list_of_results.append((line_number, line.rstrip()))
    except UnicodeDecodeError:
        pass  # silently skip files with decoding errors
    return list_of_results


def search_string_in_dir(dir_path, string_to_search):
    """Traverse dir, subdirs and files and search for the given string"""
    for subdir, dirs, files in os.walk(dir_path):
        for filename in files:
            filepath = subdir + os.sep + filename
            # Now we can search in the file
            # Added error handling here
            try:
                with open(filepath, 'r') as f:
                    if re.search(string_to_search, f.read()):
                        print(f"Found '{string_to_search}' in file {filepath}")
            except UnicodeDecodeError:
                pass  # silently skip files with decoding errors
            

search_string_in_dir("/home/exx/anaconda3/envs/rlgpu/lib/python3.7/site-packages/hydra", "env")