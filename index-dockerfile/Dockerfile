FROM python:3
ADD index.py .
RUN pip install flask
COPY index-template-directory/ .
CMD ["python", "index.py"]