"""
The module contains a set of auxiliary functions facilitating computations
"""
import yaml


def yaml_parser(input_data):
    """

    :param input_data:
    :return:
    """

    output = None

    if input_data.lower().endswith(('.yml', '.yaml')):
        with open(input_data, 'r') as stream:
            try:
                output = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        try:
            output = yaml.load(input_data)
        except yaml.YAMLError as exc:
            print(exc)

    return output
