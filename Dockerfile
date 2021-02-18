FROM ubuntu:18.04
RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN mkdir /app 
COPY requirements.txt .
COPY app_TopicModeling.py /app/app_TopicModeling.py
COPY templates /app/templates
COPY data /app/data

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN pip3 install pillow

WORKDIR /app

CMD ["python3", "app_TopicModeling.py"]


