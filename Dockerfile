FROM ubuntu:latest
RUN apt-get update -y && \
     apt-get upgrade -y && \
     apt-get dist-upgrade -y && \
     apt-get -y autoremove && \
     apt-get clean
RUN apt-get install -y build-essential python3.6 python3-pip python3-dev
RUN mkdir src
WORKDIR src/
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
RUN mkdir data
RUN apt-get install unzip 
ADD http://emodb.bilderbar.info/download/download.zip ./data/
RUN unzip ./data/download.zip -d ./data
CMD ["python3","src/misc_funcs.py"]
CMD ["./start.sh"]
