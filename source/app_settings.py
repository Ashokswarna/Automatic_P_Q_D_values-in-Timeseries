from configparser import ConfigParser
import os


def read_config(filename='config.ini', section='settings'):
    os.chdir('../config')
    parser = ConfigParser()
    parser.read(filename)
    configurations = {}

    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            configurations[item[0]] = item[1]
    else:
        raise Exception('{0} not found in {1} file'.format(section, filename))
    return configurations

