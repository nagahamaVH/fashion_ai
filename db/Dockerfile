FROM postgres:latest

RUN apt-get update
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y

RUN apt-get install --reinstall libpq-dev -y
WORKDIR /
COPY requirements.txt /
RUN pip3 install --default-timeout=100 -r requirements.txt

WORKDIR /app
# CMD python3 ./src/populate.py