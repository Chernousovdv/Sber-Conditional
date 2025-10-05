import json


def save_dict_as_json(data_dict, path, indent=4, ensure_ascii=False):
    """
    Save a dictionary as a JSON file.

    Args:
        data_dict (dict): Dictionary to save
        path (str): File path where to save the JSON
        indent (int): Indentation for pretty printing
        ensure_ascii (bool): Whether to escape non-ASCII characters
    """
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data_dict, json_file, indent=indent, ensure_ascii=ensure_ascii)
