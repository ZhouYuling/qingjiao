# 安装zookeeper
cd /root/software
tar -zxvf apache-zookeeper-3.7.1-bin.tar.gz
cd apache-zookeeper-3.7.1-bin/conf
cp zoo_sample.cfg zoo.cfg
vim zoo.cfg
dataDir=/root/software/apache-zookeeper-3.7.1-bin/data
dataLogDir=/root/software/apache-zookeeper-3.7.1-bin/log
mkdir -p /root/software/apache-zookeeper-3.7.1-bin/data
mkdir -p /root/software/apache-zookeeper-3.7.1-bin/log
vim /etc/profile
export ZOOKEEPER_HOME=/root/software/apache-zookeeper-3.7.1-bin  # 配置ZooKeeper的安装目录
export PATH=$ZOOKEEPER_HOME/bin:$PATH  # 在原PATH的基础上加入ZOOKEEPER_HOME的bin目录
source /etc/profile
zkServer.sh start
zkServer.sh status

# 安装kafka
cd /root/software/kafka_2.13-3.4.0/config
vim server.properties
broker.id=0		# 表示broker的编号，如果集群中有多个broker，则每个broker的编号需要设置的不同
listeners=PLAINTEXT://localhost:9092		# broker对外提供的服务入口地址
log.dirs=/root/software/kafka_2.13-3.4.0/kafka-logs		# Kafka存储消息日志文件的路径
zookeeper.connect=localhost:2181		# Kafka所需的ZooKeeper集群地址，本项目中ZooKeeper和Kafka都安装在本机
vim /etc/profile
export KAFKA_HOME=/root/software/kafka_2.13-3.4.0  # 配置Kafka的安装目录
export PATH=$PATH:$KAFKA_HOME/bin  # 在原PATH的基础上加入KAFKA_HOME的bin目录

kafka-server-start.sh -daemon /root/software/kafka_2.13-3.4.0/config/server.properties
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic calllog --replication-factor 1 --partitions 3
kafka-topics.sh --list --bootstrap-server localhost:9092
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic calllog --from-beginning

# 安装flume
tar -zxvf apache-flume-1.11.0-bin.tar.gz
cd /root/software/apache-flume-1.11.0-bin/conf
cp flume-env.sh.template flume-env.sh
#readlink -f /usr/bin/java
vi flume-env.sh
export JAVA_HOME=/root/software/jdk1.8.0_221
vim /etc/profile
export FLUME_HOME=/root/software/apache-flume-1.11.0-bin  # 配置Flume的安装目录
export PATH=$PATH:$FLUME_HOME/bin  # 在原PATH的基础上加入FLUME_HOME的bin目录
flume-ng help
flume-ng version

# 安装hbase
tar -zxvf hbase-2.4.15.tar.gz
cd /root/software/hbase-2.4.15/conf/
vim hbase-env.sh

<!-- 指定HBase的运行模式。 -->
<property>
  <name>hbase.cluster.distributed</name>
  <value>true</value>
</property>

<!-- 指定HBase节点在本地文件系统中的临时目录。 -->
<property>
  <name>hbase.tmp.dir</name>
  <value>./tmp</value>
</property>

<!-- 控制HBase是否检查流功能（hflush/hsync），如果您打算在rootdir表示的LocalFileSystem上运行，那就禁用此选项。 -->
<property>
  <name>hbase.unsafe.stream.capability.enforce</name>
  <value>false</value>
</property>

<!-- 指定HBase在HDFS上存储的路径，这个目录是region server的共享目录，用来持久化HBase。（不用事先创建） -->
<property>
  <name>hbase.rootdir</name>
  <value>hdfs://localhost:9000/hbase</value>
</property>

<!-- 这个是ZooKeeper配置文件zoo.cfg中的dataDir。ZooKeeper存储数据库快照的位置。 -->
<property>
  <name>hbase.zookeeper.property.dataDir</name>
  <value>/root/software/apache-zookeeper-3.7.1-bin/data</value>
</property>

vim /etc/profile
export HBASE_HOME=/root/software/hbase-2.4.15  # 配置HBase的安装目录
export PATH=$PATH:$HBASE_HOME/bin  # 在原PATH的基础上加入HBASE_HOME的bin目录

vim hbase-env.sh
export HBASE_DISABLE_HADOOP_CLASSPATH_LOOKUP="true"
start-hbase.sh 
stop-hbase.sh 
hbase shell