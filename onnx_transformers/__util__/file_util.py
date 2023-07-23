import json


def from_json(path: str):
    """
    Converts the json file at the given path to a dict.

    :param path: the path to the json file
    :return: dict
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)
