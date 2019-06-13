"""
Created on 06 Apr 2018
Project: forecastlibrary
File: request
Author: prasenjit.giri
Copyright: Accenture AI
"""


"""
{
  "forecast": {
    "data" : "filename",
    "market" : "uk",
    "category" : "Tea",
    "sku" :[
      "3100:FGB0721",
      "3100:FGB0722"],
    "forecastStart": "201722",
    "futurePeriod" : "4",
    "historicalPeriod" : "24"
  }
}

"""

class Request:
    def __init__(self, dictionary):
        dict = dictionary['forecast']
        for k, v in dict.items():
            setattr(self, k, v)
