#FROM python:3.7
FROM nvidia/cuda:10.2-base
CMD nvidia-smi


#RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
#RUN apt-get install unzip
#RUN apt-get -y install python3.7
#RUN apt-get -y install python3.7-pip

#WORKDIR /app
#ADD . /app
#RUN pip install -U pip
#RUN pip install cmake
#RUN pip install dlib
#RUN apt-get update ##[edited]
#RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN pip install -r requirements.txt
#CMD ["python", "index.py"]
