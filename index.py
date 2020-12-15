#docker build -t myapp
#docker run -d -p 5000:5000 myapp

import os, shutil
import subprocess
import requests
from flask import Flask, render_template, request, send_file
from DFDNet.test_FaceDict import process_video
import youtube_dl
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    url = request.args.get('url')
    time1 = request.args.get('timecod1')
    time2 = request.args.get('timecod2')

    file_path="app/DFDNet/test_FaceDict.py"
    #file_path="test.py "+url+" "+time1
    #os.system(f'py {file_path}')
    #requests.post('http://localhost:4000/', data={'source_url': url, 'target_start': time1, 'target_end': time2})
    process_video(url, time1, time2)
    return render_template('form2.html')

@app.route('/DFDNet/Results/TestVideoResults/Step1_Cropping/crop_downloaded_video.mp4', methods=['GET', 'POST'])
def video_result():
    return send_file('DFDNet/Results/TestVideoResults/Step1_Cropping/crop_downloaded_video.mp4', as_attachment=True)

@app.route('/result', methods=['POST'])
def result():
    #subprocess.call("app/templates/video.html", shell=True)
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
