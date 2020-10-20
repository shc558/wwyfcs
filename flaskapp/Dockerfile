FROM python:3.8-slim
RUN apt update
WORKDIR /root/app
ADD requirements.txt /root/app/requirements.txt
RUN pip install -r /root/app/requirements.txt
ADD . /root/app
ENV PORT 8080
CMD ["gunicorn", "app:app", "--config=config.py"]
