FROM python:3
ADD test_FaceDict.py /
RUN pip install dlib opencv-python tqdm scikit-image dominate
COPY pivot-neuron-directory/ .
CMD ["python", "test_FaceDict.py"]