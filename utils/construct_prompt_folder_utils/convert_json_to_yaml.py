
import os
import json
import yaml
import shutil
import glob


from .make_dir_if_it_doesnt_exist_then_return_them import make_dir_if_it_doesnt_exist_then_return_them

def convert_json_to_yaml(prompts_dir: str) -> None:
    """
    Example:
    >>> json_directory = "../prompts/"
    >>> convert_json_to_yaml(json_directory)
    """
    _, json_dir, yaml_dir = make_dir_if_it_doesnt_exist_then_return_them(prompts_dir)

    # Convert each text file to JSONL and YAML
    counter = 0
    for filename in os.listdir(json_dir):
        if filename.endswith(".jsonl"):
            # Define the filepaths
            json_path = os.path.join(json_dir, filename)
            yaml_path = os.path.join(yaml_dir, filename[:-5] + ".yaml")

            if os.path.exists(yaml_path):
                pass
            else:
                # Get the JSON data
                print(f"Converting {filename} to yaml...")
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)

                # Format it as YAML
                yaml_content = []
                for item in json_data:
                    yaml_content.append(f"- role: {item['role']}")
                    yaml_content.append(f"  content: |")
                    content = item['content'].replace('\\n', '\n')
                    content = content.replace('\\"', '"')
                    for line in content.split('\n'):
                        yaml_content.append(f"    {line}")

                with open(yaml_path, "w") as yaml_file:
                    yaml_file.write('\n'.join(yaml_content))

            counter += 1
    print(f"Converted {counter} jsonl files to yaml.")
    return


def convert_text_to_single_jsonl(prompts_dir: str) -> None:
    """
    Example:
    >>> txt_directory = "../prompts/"
    >>> convert_text_to_single_jsonl(txt_directory)
    """
    txt_dir, json_dir, _ = make_dir_if_it_doesnt_exist_then_return_them(prompts_dir)

    counter = 0
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            # Define the directories for .txt and .jsonl files
            txt_path = os.path.join(txt_dir, filename)
            jsonl_path = os.path.join(json_dir, filename[:-4] + ".jsonl")

            if os.path.exists(jsonl_path):
                pass
            else:
                print(f"Converting {filename} to jsonl...")
            
                with open(txt_path, "r") as txt_file:
                    # Read the entire content of the file, preserving whitespace
                    file_content = txt_file.read()
                    # Create a dictionary with the entire file content
                    json_obj = {"text": file_content}
        
                with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
                    # Write the single JSON object to the .jsonl file
                    jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            counter += 1
    print(f"Converted {counter} jsonl files to yaml.")
    return

