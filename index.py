# -*- coding: utf-8 -*-

import requests
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/form', methods=['GET', 'POST'])
def form():
    url = request.args.get('url')
    time1 = request.args.get('timecod1')
    time2 = request.args.get('timecod2')
    requests.post('http://0.0.0.0:4000', data={'source_url': url, 'target_start': time1, 'target_end': time2})
    return render_template('form2.html')


@app.route('/result', methods=['POST'])
def result():

    return render_template('video.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
