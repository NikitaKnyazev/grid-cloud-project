FROM python:3
ADD test_FaceDict.py /
ADD form.py /
RUN pip install numpy dlib opencv-python tqdm scikit-image dominate
COPY pivot-neuron-directory/ .
CMD ["python", "form.py"]