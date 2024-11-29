#步骤一创建agent配置文件
cd $FLUME_HOME
mkdir -p jobs/calllog_kafka.conf
vim calllog_kafka.conf


#任务二：使用制定采集方案启动FLUME
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic calllog --from-beginning

## 前台启动运行
bin/flume-ng agent -c conf/ -f jobs/calllog_kafka.conf -n a1 -Dflume.root.logger=INFO,console
# 或者
bin/flume-ng agent --conf conf/ --conf-file jobs/calllog_kafka.conf --name a1 -Dflume.root.logger=INFO,console
## 后台启动运行（提前创建日志文件所在父目录）
nohup bin/flume-ng agent -c conf/ -f jobs/calllog_kafka.conf -n a1 -Dflume.root.logger=INFO,console >/root/bigdata/project3/logs/calllog_kafka.log 2>&1 &
# 或者
nohup bin/flume-ng agent --conf conf/ --conf-file jobs/calllog_kafka.conf --name a1 -Dflume.root.logger=INFO,console >/root/bigdata/project3/logs/calllog_kafka.log 2>&1 &


#运行生产数据脚本呢
sh /root/bigdata/project3/data/produceLog.sh


