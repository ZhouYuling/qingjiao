
27.19.74.143 - - [30/May/2013:17:38:20 +0800] "GET /static/image/common/faq.gif HTTP/1.1" 200 1127
110.52.250.126 - - [30/May/2013:17:38:20 +0800] "GET /data/cache/style_1_widthauto.css?y7a HTTP/1.1" 200 1292
27.19.74.143 - - [30/May/2013:17:38:20 +0800] "GET /static/image/common/hot_1.gif HTTP/1.1" 200 680
27.19.74.143 - - [30/May/2013:17:38:20 +0800] "GET /static/image/common/hot_2.gif HTTP/1.1" 200 682
27.19.74.143 - - [30/May/2013:17:38:20 +0800] "GET /static/image/filetype/common.gif HTTP/1.1" 200 90
110.52.250.126 - - [30/May/2013:17:38:20 +0800] "GET /source/plugin/wsh_wx/img/wsh_zk.css HTTP/1.1" 200 1482
110.52.250.126 - - [30/May/2013:17:38:20 +0800] "GET /data/cache/style_1_forum_index.css?y7a HTTP/1.1" 200 2331
110.52.250.126 - - [30/May/2013:17:38:20 +0800] "GET /source/plugin/wsh_wx/img/wx_jqr.gif HTTP/1.1" 200 1770
27.19.74.143 - - [30/May/2013:17:38:20 +0800] "GET /static/image/common/recommend_1.gif HTTP/1.1" 200 1030
110.52.250.126 - - [30/May/2013:17:38:20 +0800] "GET /static/image/common/logo.png HTTP/1.1" 200 4542
27.19.74.143 - - [30/May/2013:17:38:20 +0800] "GET /data/attachment/common/c8/common_2_verify_icon.png HTTP/1.1" 200 582
110.52.250.126 - - [30/May/2013:17:38:20 +0800] "GET /static/js/logging.js?y7a HTTP/1.1" 200 603
8.35.201.144 - - [30/May/2013:17:38:20 +0800] "GET /uc_server/avatar.php?uid=29331&size=middle HTTP/1.1" 301 -
...


## 定义 sources、channels 以及 sinks
agent1.sources = logSrc
agent1.channels = fileChannel
agent1.sinks = hdfsSink

## logSrc 的配置
agent1.sources.logSrc.type = exec
agent1.sources.logSrc.command = tail -F /home/hadoop-twq/spark-course/steaming/flume-course/demo3/logs/webserver.log

## hdfsSink 的配置
agent1.sinks.hdfsSink.type = hdfs
agent1.sinks.hdfsSink.hdfs.path = hdfs://master:9999/user/hadoop-twq/spark-course/steaming/flume/%y-%m-%d
agent1.sinks.hdfsSink.hdfs.batchSize = 5
agent1.sinks.hdfsSink.hdfs.useLocalTimeStamp = true

## fileChannel 的配置
agent1.channels.fileChannel.type = file
agent1.channels.fileChannel.checkpointDir = /home/hadoop-twq/spark-course/steaming/flume-course/demo2-2/checkpoint
agent1.channels.fileChannel.dataDirs = /home/hadoop-twq/spark-course/steaming/flume-course/demo2-2/data

## 通过 fileChannel 连接 logSrc 和 hdfsSink
agent1.sources.logSrc.channels = fileChannel
agent1.sinks.hdfsSink.channel = fileChannel


bin/flume-ng agent -n a1 -c conf -f conf/配置文件名  -Dflume.root.logger=INFO,console


# 创建Kafka主题logtopic，设置副本数为1，分区数为3。
/root/software/kafka/bin/kafka-topics.sh --create --zookeeper ???:???--replication-factor ??? --partitions ??? --topic ???


# Mysql数据库表创建
# 登录Mysql 密码123456
mysql -u root -p
create database ???;


# 进入hongya数据库
use hongya;
# 创建表
CREATE TABLE `logpv` (
  `id` INT(4) NOT NULL AUTO_INCREMENT,
  `pvcount` VARCHAR(20) DEFAULT NULL,
  `time` VARCHAR(30) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MYISAM DEFAULT CHARSET=utf8;


CREATE TABLE `logjumper` (
  `id` INT(4) NOT NULL AUTO_INCREMENT,
  `jumper` VARCHAR(20) DEFAULT NULL,
  `time` VARCHAR(30) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MYISAM DEFAULT CHARSET=utf8;


CREATE TABLE `logreguser` (
  `id` INT(4) NOT NULL AUTO_INCREMENT,
  `reguser` VARCHAR(20) DEFAULT NULL,
  `time` VARCHAR(30) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MYISAM DEFAULT CHARSET=utf8;




