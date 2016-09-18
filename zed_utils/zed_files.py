import os

def create_dir_if_needed(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)