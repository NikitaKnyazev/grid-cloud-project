FROM python:3.7
WORKDIR /app
ADD . /app
RUN pip install -U pip
RUN pip install cmake
RUN pip install dlib
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
CMD ["python", "index.py"]
