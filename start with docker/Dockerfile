FROM python:3.7
WORKDIR /dfdnet
ADD . /dfdnet
RUN pip install -U pip
RUN pip install -r requirements.txt
CMD ["python", "index.py"]
