FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

RUN apt-get update
RUN apt-get install python3 python3-pip -y

RUN python3 -m pip install pip -U

WORKDIR /
COPY requirements.txt /
RUN pip3 install --default-timeout=100 -r requirements.txt

COPY ./download_fasterrcnn.py /
RUN python3 download_fasterrcnn.py

WORKDIR /app

CMD python3 ./src/api.py