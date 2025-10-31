#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os

def write_string_to_file(string: str, output_file: str):
    """
    Write a given string to a file
    """
    with open(output_file, "w") as f_out:
        f_out.write(string)

def check_if_file_exists(file_path: str):
    """
    Check if a file exists
    """
    return os.path.exists(file_path)

def open_file_contents(file_path: str):
    with open(file_path) as f:
        return f.read()
        
def make_dirs(dir_path: str):
    """
    Create a directory
    """
    os.makedirs(dir_path, exist_ok=True)