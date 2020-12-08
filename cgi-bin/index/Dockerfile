FROM python:3
ADD index.py .
RUN pip install os
COPY index-template-directory/ .
CMD ["python", "index.py"]