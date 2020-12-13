# -*- coding: utf-8 -*-
#docker build -t myapp
#docker run -d -p 5000:5000 myapp

import os, shutil
import subprocess
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    '''
    path = os.getcwd()+'/Results/TestVideoResults/'
    for dir in os.listdir(path):
        for file in os.listdir(path+'/'+dir):
            os.remove(path+'/'+dir+'/'+file)
            #print(path+'/'+dir+'/'+file)
            '''
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    url = request.args.get('url')
    time1 = request.args.get('timecod1')
    time2 = request.args.get('timecod2')

    #file_path=os.getcwd()+"/test_FaceDict.py --gpu_ids -1 --source_url "+url+" --start "+time1+" --stop "+time2
    #file_path="test.py "+url+" "+time1
    #os.system(f'py {file_path}')
    return render_template('form2.html')

@app.route('/result', methods=['POST'])
def result():
    #subprocess.call("app/templates/video.html", shell=True)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
