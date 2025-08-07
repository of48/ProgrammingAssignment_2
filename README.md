Programming Assignment 2
Wine Quality Prediction ML model in Spark over AWS
# Assignment Goals
- Develop parallel machine learning (ML) applications in Amazon AWS cloud platform
- how to use Apache Spark to train an ML model in parallel on multiple EC2 instances
- how to use Spark’s MLlib to develop and use an ML model in the cloud
- How to use Docker to create a container for your ML model to simplify model deployment
# Description
Build a wine quality prediction ML model in Spark over AWS. The model must be trained in parallel on multiple EC2 instances. Then, you need to save and load the model in an application that will perform wine quality prediction; this application will run on one EC2 instance. The assignment must be implemented in Java, Scala, or Python on Ubuntu Linux.
Input for model training: we share 2 datasets with you for your ML model. Each row in a dataset is for one specific wine, and it contains some physical parameters of the wine as well as a quality score.
- TrainingDataset.csv: you will use this dataset to train the model in parallel on multiple EC2 instances.
- ValidationDataset.csv: you will use this dataset to validate the model and optimize its performance (i.e., select the best values for the model parameters).
# EC Instances
Create 5 EC2s, 4 Spark EC2s and one for the Docker & Network File System (NFS) share.
Select the following image “Ubuntu Server 24.04 LTS (HVM)”, create t3.large for each Spark machine and one t3.medium for SparkPredictiorApp to run the docker one and the NFS share.
# EC2 Security Group
Allow Inbound traffic to access the following ports TCP 22, UDP 2049, TCP 2049, TCP 7077, TCP 8080 and ICMP.
- SSH Port 22 for Remote Access
- TCP/UDP 2049 for Network File Service
- TCP 7077 for Spark Master to communicate with workers and applications using RPC (Remote Procedure Call) communication.
- TPC 8080 to access Spark Master web interface
- ICMP to make sure that all EC2s can communicate with each other.
# Network File System (NFS) File System Share
Create an NFS share on the SparkPredictorApp EC2 and share it with all other EC2s (SPARK-Master, SPARK-Worker-1, SPARK-Worker-2, SPARK-Worker-3) to read the datasets and write the model to.
All machines are mounted to “/mnt/nfs_project” to read the TrainingDataset.csv & ValidationDataset.csv and for the Spark-Master to write the model.
- Install NFS Kernel Server Steps
- sudo apt install nfs-kernel-server -y
- Directory Creation
- sudo mkdir -p /mnt/nfs_project
- Changing user and group owners
- sudo chown nobody:nogroup /mnt/nfs_project
- Set directory permissions to read, write and execute
- sudo chmod 777 /mnt/nfs_project
- Client Permissions
- sudo nano /etc/exports
- Grant permissions to the Servers
- /mnt/nfs_project *(rw,sync,no_subtree_check)(rw,sync,no_subtree_check)
- Export Shared Directory
- sudo exportfs -a
- Restart the NFS Kernel Server
- sudo systemctl restart nfs-kernel-server
- Allow Clients Through AWS Security Groups Ports 2049 TCP & UDP
- Set up a Mount Point on the Client EC2s
- sudo mount 172.31.82.197:/mnt/nfs_project /mnt/nfs_project
# EC2 Environment Setup
Setting up the applications on all machines
## Application Versions
- Java Runtime (OpenJDK 11.0.28)
- Apache Hadoop (3.3.6)
- Apache Spark (3.5.6)
- Python 3.12.3
### Java 11.0.28
- sudo apt install openjdk-11-jdk -y
### Hadoop 3.3.6
- wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
- sudo tar -xvzf hadoop-3.3.6.tar.gz -C /opt/
- sudo mv /opt/hadoop-3.3.6 /opt/Hadoop
### Spark 3.5.6
- wget https://downloads.apache.org/spark/spark-3.5.6/spark-3.5.6-bin-hadoop3.tgz
- sudo tar -xvzf spark-3.5.6-bin-hadoop3.tgz -C /opt/
- sudo mv /opt/spark-3.5.6-bin-hadoop3 /opt/spark
On the SparkPredictorApp machine
### Docker
- sudo apt install docker.io -y
- sudo systemctl start docker
- sudo systemctl enable docker
## Environment variables
Updating the environment variables, access home directory
cd ~
nano .bashrc
- Java environment
- export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
- export PATH=$JAVA_HOME/bin:$PATH
- Python virtualenv for PySpark
- export PYSPARK_PYTHON=/home/ubuntu/myenv/bin/python
- export PYSPARK_DRIVER_PYTHON=/home/ubuntu/myenv/bin/python
- Hadoop environment
- export HADOOP_HOME=/opt/hadoop
- export HADOOP_COMMON_HOME=/opt/hadoop
- export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
- export PATH=$PATH:$HADOOP_HOME/bin
- export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH
- Spark environment
- export SPARK_HOME=/opt/spark
- export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
source .bashrc (to save the environment variables)
# Model Training
### Spark-Master EC2
#### Python Installation
sudo apt install python3-pip python3-venv -y
python3 -m venv ~/myenv
source ~/myenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy
python3 -m pip install pyspark
- spark-submit train_model.py
- After running the training model the result for Logistic Regression Model is 5672726692311375 and Random Forest Classifier Model is 0.5113001410286163
- Logistic_Regression_Model F1 Score: 0.5672726692311375
- Random_Forest_Classifier_Model F1 Score: 0.5113001410286163
- Model output is saved in the /mnt/nfs/model
### Apache Spark Application Web Interface
Logging to the Spark Master Web Interface
- Summary of the Cluster status, total number of available cores and memory and the number of active workers http://spark-master-ip:8080/
- Running jobs http://spark-master-ip:4040/jobs/
### Docker
mkdir wine-predictor
docker login -u
type in credentaisl
cd wine-predictor
docker build -t username /wine-predictor .
docker push username/wine-predictor
docker run -v /mnt/nfs_project:/mnt/nfs_project username/wine-predictor /mnt/nfs_project/ValidationDataset.csv
