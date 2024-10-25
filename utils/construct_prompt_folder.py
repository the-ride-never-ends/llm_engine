
import os

from .construct_prompt_folder_utils.convert_text_to_single_jsonl import convert_text_to_single_jsonl
from .construct_prompt_folder_utils.convert_json_to_yaml import convert_json_to_yaml
from .construct_prompt_folder_utils.convert_escaped_newlines_in_yaml import convert_escaped_newlines_in_yaml


def construct_prompt_folder(prompt_folder_path: str) -> None:
    if not os.path.exists(prompt_folder_path):
        os.makedirs(prompt_folder_path)
        return
    else:
        convert_text_to_single_jsonl(prompt_folder_path)
        convert_json_to_yaml(prompt_folder_path)




