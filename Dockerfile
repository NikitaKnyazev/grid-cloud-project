FROM python:3.7
WORKDIR /app
ADD . /app
RUN pip install -U pip
RUN pip install cmake
RUN pip install dlib
RUN pip install torch==1.7.1
RUN pip install torchvision
RUN pip install -r requirements.txt
CMD ["python", "index.py"]
