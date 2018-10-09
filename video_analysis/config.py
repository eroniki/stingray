import json


class config_parser(object):
    """docstring for config_parser."""

    def __init__(self, fname):
        super(config_parser, self).__init__()
        self.fname = fname

    def parse(self):
        with open(self.fname, "r") as read_file:
            data = json.load(read_file)
        return data
