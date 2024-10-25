import os

def make_dir_if_it_doesnt_exist_then_return_them(prompts_dir: str) -> tuple[str, str, str]:
    path_list = []
    for subdir in ["text", "jsonl", "yaml"]:
        path = os.path.join(prompts_dir, subdir)
        # Make the directory if it doesn't exist
        if not os.path.exists(path):
            os.mkdir(path)
        else: # If it does exist, make sure there's something in them.
            if not any(filename.endswith(f".{subdir}") for filename in os.listdir(path)):
                raise Exception(f"{subdir} is empty.")
        path_list.append(path)
    return tuple(path_list)
