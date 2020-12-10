FROM python:3
COPY . /app
WORKDIR /app
RUN pip install flask \
#&& pip install numpy \
#&& pip install torch \
#&& pip install torchvision \
&& pip install Pillow \
#&& pip install dominate \
&& pip install youtube_dl \
&& pip install moviepy
#&& pip install opencv-python \
#&& pip install tqdm \
#&& pip install cmake \
#&& pip install dlib \
#&& pip install scikit-image
CMD ["python", "index.py"]
