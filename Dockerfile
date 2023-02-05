FROM ubuntu:20.04

ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT 5000
COPY requirements.txt ./requirements.txt
RUN apt-get update
RUN apt-get -y install python3-pip
RUN pip install -r requirements.txt

COPY model_creation.py ./model_creation.py
COPY preprocessing.py  ./preprocessing.py
COPY train.py ./train.py

#RUN python3 train.py
EXPOSE 5000

CMD ["mlflow", "run", ".", "-e", "train.py", "--no-conda", "--host", "0.0.0.0", "--port", "5000"]