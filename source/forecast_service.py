"""
Created on 3 Mar 2018
Project: forecastlibrary
File: forecast_service
Author: prasenjit.giri
Copyright: Accenture AI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from forecast import forecast_oot
from app_settings import read_config
from forecast_service_helper import parse_json

app = Flask(__name__)
CORS(app)


@app.route('/df/v1/forecast', methods=['POST'])
@cross_origin()
def forecast():
    json_string = request.get_json()
    req_dict = parse_json(json_string)
    result = forecast_oot(req_dict)
    return jsonify(result)


if __name__ == '__main__':
    app_settings = read_config()
    port = int(app_settings['port'])
    app.run(port=port, debug=True)
