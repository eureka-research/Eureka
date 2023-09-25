
def inject_code(task_code, reward_code, fill_label):
    lines = task_code.split("\n")
    for line in lines:
        if fill_label in line:
            num_spaces = len(line) - len(line.lstrip())
            reward_code = indent_code(reward_code, num_spaces)
            break
    return task_code.replace(fill_label, fill_label + "\n" + reward_code)

def indent_code(code, num_spaces):
    return "\n".join([" " * num_spaces + line for line in code.split("\n")])