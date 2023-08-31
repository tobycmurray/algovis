# syntax=docker/dockerfile:1

FROM python:3.11

RUN apt install -y make

RUN pip3 install jupyterlab matplotlib networkx

WORKDIR /python-docker

COPY . .

CMD [ "make" ]
