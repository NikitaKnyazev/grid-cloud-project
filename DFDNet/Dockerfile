FROM python:3.7

WORKDIR /app/DFDNet
ADD . /app/DFDNet

RUN pip install -r requirements.txt

CMD ["python", "test_FaceDict.py --gpu_ids -1 --source_url "+url+" --start "+time1+" --stop "+time2"]
