# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, make_response

from test_FaceDict import process_video

app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    data = request.form
    source_url = data['source_url']
    target_start = data['target_start']
    target_end = data['target_end']
    process_video(source_url, target_start, target_end)

    return make_response()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
