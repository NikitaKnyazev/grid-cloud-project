# -*- coding: utf-8 -*-
import os, shutil
from flask import Flask, render_template, request, send_file
from DFDNet.test_FaceDict import process_video
#import youtube_dl
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    path = os.getcwd()+'/DFDNet/Results/TestVideoResults/'
    for dir in os.listdir(path):
        for file in os.listdir(path+'/'+dir):
            os.remove(path+'/'+dir+'/'+file)
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    url = request.args.get('url')
    time1 = request.args.get('timecod1')
    time2 = request.args.get('timecod2')
    process_video(url, time1, time2)
    return render_template('form2.html')

@app.route('/DFDNet/Results/TestVideoResults/Step6_FinalVideo/result.mp4', methods=['GET', 'POST'])
def video_result():
    return send_file('DFDNet/Results/TestVideoResults/Step6_FinalVideo/result.mp4', as_attachment=True)

@app.route('/templates/video.png', methods=['GET', 'POST'])
def img():
    return send_file('templates/video.png', as_attachment=True)

@app.route('/result', methods=['GET','POST'])
def result():
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
