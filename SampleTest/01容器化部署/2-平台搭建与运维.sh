#一、配置基础环境
#1.更新主机名，配置hosts文件，以node01为准同步时间
docker ps -a
docker rm ID

docker run -it --name node01 -h node01 -d centos7_mysql:0.0.1
docker run -it --name node02 -h node02 -d centos7_mysql:0.0.1
docker run -it --name node03 -h node03 -d centos7_mysql:0.0.1

#新建三个窗口链接这三台机器
docker exec -it node01 /bin/bash
docker exec -it node02 /bin/bash
docker exec -it node03 /bin/bash

#配置hosts，三台机器上都要配置
vim /etc/hosts
ip1 node01
ip2 node02
ip3 node03

#配置node01为服务器
vim /etc/ntp.conf
#注释server 0.centos.pool.ntp.org iburst
server 127.127.1.1
fudge 127.127.1.0 stratum 10

#2.执行命令生成公钥、私钥，实现三台机器间的免密登录
ssh-keygen
#回车
#y
#回车
cat /root/.ssh/.ssh/id_rsa.pub
vim /root/.ssh/authorized_keys
#添加到最后
#node02和node03执行vim /root/.ssh/authorized_keys进行同样操作
#在node01上执行免密登录
ssh node02
ssh node03

#3.安装jdk到/root/software下
cp /root/software/package/jdk-8u212-linux-x64.tar.gz /root/software
tar -zxvf jdk-8u212-linux-x64.tar.gz

#4.在环境变量中配置jdk，查看jdk的版本
vim /etc/profile
export JAVA_HOME=/root/software/jdk1.8.0_212/
export PATH=$PATH:$JAVA_HOME/bin

source /etc/profile
java -version
scp -r /etc/profile node02:/etc/
scp -r /etc/profile node03:/etc/
scp -r /root/software node02:/root/
scp -r /root/software node03:/root/
#node02和node03执行环境生效
source /etc/profile

#二、Hadoop完全分布式安装配置
#1.将hadoop安装包解压到/root/software目录下
cd /root/
tar -zxvf hadoop-3.1.3.tar.gz -C ./
cd hadoop-3.1.3/etc/hadoop/
vim hadoop-env.sh
export JAVA_HOME=/root/software/jdk1.8.0_212/

vim core-site.xml
<configuration>
<property>
<name>fs.defaultFS</name>
<value>hdfs://node01:9000</value>
<property>
<property>
<name>hadoop.tmp.dir</name>
<value>/root/software/hadoop-3.1.3/data</value>
<property>
</configuration>

vim hdfs-site.xml
<configuration>
<property>
<name>dfs.namenode.http-address</name>
<value>node01:9870</value>
<property>
<property>
<name>dfs.replication</name>
<value>3</value>
<property>
</configuration>

vim mapred-site.xml
<configuration>
<property>
<name>mapreduce.framework.name</name>
<value>yarn</value>
<property>
<property>
<name>yarn.app.mapreduce.am.env</name>
<value>HADOOP_MAPRED_HOME=/root/software/hadoop-3.1.3</value>
<property>
<property>
<name>mapreduce.map.env</name>
<value>HADOOP_MAPRED_HOME=/root/software/hadoop-3.1.3</value>
<property>
<property>
<name>mapreduce.reduce.env</name>
<value>HADOOP_MAPRED_HOME=/root/software/hadoop-3.1.3</value>
<property>
</configuration>

vim yarn-site.xml
<configuration>
<property>
<name>yarn.nodemanger.aux-services</name>
<value>mapreduce_shuffle</value>
<property>
<property>
<name>yarn.resourcemanager.hostname</name>
<value>node01</value>
<property>
<property>
<name>yarn.resourcemanger.address</name>
<value>node01:8032</value>
<property>
<property>
<name>yarn.resourcemanger.scheduler.address</name>
<value>node01:8030</value>
<property>
<property>
<name>yarn.nodemanger.env-whitelist</name>
<value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_MAPRED_HOME</value>
<property>
</configuration>

vim workers
node01
node02
node03

scp -r /root/software/hadoop-3.1.3 root@node02:/root/software
scp -r /root/software/hadoop-3.1.3 root@node03:/root/software

vim /etc/hosts
export HADOOP_HOME=/root/software/hadoop-3.1.3
export PATH=$PATH:$HAOOP_HOME/bin:$HADOOP_HOME/sbin
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export HDFS_JOURNALNODE_USER=root
export HDFS_ZKFC_USER=root

scp -r /etc/profile root@node02:/etc/
scp -r /etc/profile root@node03:/etc/
#node01 node02 node03上执行
source /etc/profile

#2.开启集群，查看各节点进程
hadoop namenode -format
start-all.sh
jps


#三、Hive安装配置
#1.将hive解压到/root/software目录下
docker cp /root/software/package/apache-hive-3.1.2-bin.tar.gz node03:/root/software/
docker cp /root/software/package/mysql-connector-5.1.37-bin.jar node03:/root/software/
#node03上操作
cd /root/software
tar -zxvf apache-hive-3.1.2-bin.tar.gz -C ./
vim /etc/profile
export HIVE_HOME=/root/software/apache-hive-3.1.2-bin
export PATH=$PATH:$HIVE_HOME/bin:$HIVE_HOME/sbin
#配置hive相关配置
cd apache-hive-3.1.2-bin/conf/
cp hive-env.sh.template hive-env.sh
vim hive-env.sh
export HADOOP_HOME=/root/software/hadoop-3.1.3
export HIVE_CONF_DIR=/root/software/apache-hive-3.1.2-bin/conf
export HIVE_AUX_JARS_PATH=/root/software/apache-hive-3.1.2-bin/lib

vim hive-site.xml
<configuration>
<property>
<name>javx.jdo.option.ConnectionURL</name>
<value>jdbc:mysql://node03:3306/hivedb?createDatabaseIfNotExist=true&amp;useSSL=false&amp;useUnicode=true&amp;characterEncoding=UTF-8</value>
<property>
<property>
<name>javax.jdo.option.ConnectionDriverName</name>
<value>com.mysql.jdbc.Driver</value>
<property>
<property>
<name>javax.jdo.option.ConnectionUserName</name>
<value>root</value>
<property>
<property>
<name>javax.jdo.option.ConnectionPassword</name>
<value>123456</value>
<property>
</configuration>

cp /root/sofware/mysql-connector-5.1.37-bin.jar /root/software/apache-hive-3.1.2-bin/lib
schematool -dbType mysql -initSchema
#报错，应该是mysql没有配置

#启动mysql
/usr/sbin/mysqld --user=mysql &
mysql -uroot -p123456
use mysql;
select user,host from user;
update user set host='%' where host='localhost';
flush privileges;
exit

schematool -dbType mysql -initSchema
#schematool completed表示成功

#进入hive创建一个库
hive
create database hive;
docker exec node03 /bin/bash -c '/root/software/hadoop-3.1.3/bin/hdfs dfs -ls -R /'

#四、Flume安装配置
#1.解压Flume1.11.0解压到/root/software目录下
tar -zxvf apache-flume-1.11.0-bin.tar.gz


#2.在conf路径下，配置flume-env.sh文件，并生效文件，查看是否安装成功
cd /root/software/apache-flume-1.11.0-bin/conf
cp flume-env.sh.template flume-env.sh
vi flume-env.sh
export JAVA_HOME=/root/software/jdk1.8.0_221
vim /etc/profile
export FLUME_HOME=/root/software/apache-flume-1.11.0-bin  # 配置Flume的安装目录
export PATH=$PATH:$FLUME_HOME/bin  # 在原PATH的基础上加入FLUME_HOME的bin目录
flume-ng help
flume-ng version

#保证验证通过
/root/software/apache-flume-1.11.0-bin/bin/flume-ng version >> /root/flumeversion.txt
