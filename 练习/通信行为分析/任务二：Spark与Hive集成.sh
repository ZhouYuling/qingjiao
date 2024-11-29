
#hive连接
hive
hive --service cli
#创建测试表
create table student(id int,name string);
#插入测试数据
insert overwrite table student values("1","shiny");



#任务2：Spark安装及环境配置（Spark on YARN模式）
cd /root/software/		# 进入目录
tar -zxvf spark-3.0.0-bin-without-hadoop.tgz	# 解压安装包
ln -s spark-3.0.0-bin-without-hadoop spark	# 创建软链接

vim /etc/profile
# 配置Spark的安装目录
export SPARK_HOME=/root/software/spark
# 在原PATH的基础上加入Spark的bin目录
export PATH=$PATH:$SPARK_HOME/bin


source /etc/profile


cd $SPARK_HOME/conf	# 进入目录
cp spark-env.sh.template spark-env.sh	# 复制并重命名


vim spark-env.sh
# Spark on YARN配置
# Hadoop集群（客户端）配置文件的目录，读取HDFS上文件和运行Spark在YARN集群时需要
export HADOOP_CONF_DIR=/root/software/hadoop-3.3.3/etc/hadoop
export YARN_CONF_DIR=/root/software/hadoop-3.3.3/etc/hadoop
# Spark的classpath依赖配置，设置为Hadoop命令路径
export SPARK_DIST_CLASSPATH=$(/root/software/hadoop-3.3.3/bin/hadoop classpath)

#配置历史服务器
cd $SPARK_HOME/conf	# 进入目录
cp spark-defaults.conf.template spark-defaults.conf	# 复制并重命名
vim spark-defaults.conf

# 默认提交到YARN集群运行
spark.master                     yarn
# 配置日志存储路径，HDFS上的目录需要提前创建
spark.eventLog.enabled           true
spark.eventLog.dir               hdfs://localhost:9000/spark/log
# Executor和Driver堆内存
spark.executor.memory            2g
spark.driver.memory              2g


#创建hdfs目录
hadoop fs -mkdir -p /spark/log


# 提交示例程序
spark-submit \
--class org.apache.spark.examples.SparkPi \
--master yarn \
examples/jars/spark-examples_2.12-3.0.0.jar \
10



#任务3：Hive on Spark安装部署
#1. 上传 Spark 依赖到 HDFS
hadoop fs -mkdir /spark/jars	# 创建jar包存放目录
cd $SPARK_HOME/jars	# 进入目录
hadoop fs -put * /spark/jars	# 上传jar包，注意这里上传的是Spark纯净的jar包，不包含Hadoop的jar包


#配置hive-site.xml
#<!--Spark依赖位置（注意：端口号9000必须和NameNode的端口号一致）-->
#<property>
#<name>spark.yarn.jars</name>
#<value>hdfs://localhost:9000/spark/jars/*</value>
#</property>
#<!--Hive执行引擎，可以是mr、tez或者spark，默认值为mr-->
#<property>
#<name>hive.execution.engine</name>
#<value>spark</value>
#</property>
#<!--Hive和Spark连接超时时间，默认值为1000ms-->
#<property>
#<name>hive.spark.client.connect.timeout</name>
#<value>10000ms</value>
#</property>



