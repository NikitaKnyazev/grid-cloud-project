# -*- coding: utf-8 -*-
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    f = open('index.html', 'r', encoding='UTF-8')
    s = f.read()
    f.close()
    print('Content-type: text/html\n')
    return s

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
