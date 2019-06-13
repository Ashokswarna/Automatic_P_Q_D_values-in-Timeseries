"""
Created on 06 Apr 2018
Project: forecastlibrary
File: forecast_service_helper
Author: prasenjit.giri
Copyright: Accenture AI
"""

from request import Request


def parse_json(parsed_json):
    """
    Parses json string to dictionary
    :param parsed_json: JSON string
    :return: Dictionary
    """
    req_obj = Request(parsed_json)
    return req_obj
