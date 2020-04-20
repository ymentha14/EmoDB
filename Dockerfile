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
RUN mkdir data
RUN apt-get install unzip 
RUN yes|apt-get install libsndfile1
ADD http://emodb.bilderbar.info/download/download.zip ./data/
RUN unzip ./data/download.zip -d ./data
COPY . .
CMD ["python3","src/misc_funcs.py"]
CMD ["./start.sh"]
