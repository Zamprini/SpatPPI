import os


def extract_prefixes_from_folder(folder_path):
    file_prefixes = []
    for file_name in os.listdir(folder_path):
        # Get the prefix of the filename
        prefix = file_name.split('.')[0]
        file_prefixes.append(prefix)
    return file_prefixes