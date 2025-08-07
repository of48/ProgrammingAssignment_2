# Ubuntu image
FROM ubuntu:24.04

# Install packages and clean it
RUN apt-get update && apt-get install -y openjdk-11-jdk python3 python3-pip curl wget bash tar && apt-get clean


# Install dependecies
RUN pip install --break-system-packages pyspark==3.5.6 numpy

# Configure env variables 
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$JAVA_HOME/bin:$SPARK_HOME/bin

# Download & install Apache Spark
RUN wget https://archive.apache.org/dist/spark/spark-3.5.6/spark-3.5.6-bin-hadoop3.tgz && tar -xzf spark-3.5.6-bin-hadoop3.tgz &&  mv spark-3.5.6-bin-hadoop3 /opt/spark && rm spark-3.5.6-bin-hadoop3.tgz

COPY predict_app.py /app/predict_app.py

WORKDIR /app

ENTRYPOINT ["python3", "predict_app.py"]

