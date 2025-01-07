import os


def search_folders(directory, search_term):
    """
    Function to search directories for files/directories with keyword or phrases in their file names.
    Only works if only one item has the key-phrase.
    :param directory: path to the directory to be searched
    :param search_term: key-phrase string
    :return: if a file is found it will be returned, else None will be returned
    """
    if not search_term:
        return None

    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if search_term in dir_name:
                result = os.path.join(root, dir_name)
                return result
