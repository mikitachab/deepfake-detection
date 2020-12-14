import json


def load_json(json_path):
    with open(json_path) as file:
        return json.load(file)
