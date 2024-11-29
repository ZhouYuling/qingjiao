#任务1：添加ETL拦截器
cd /root/bigdata/data/project2/soft/
cp my-interceptor-etl.jar /root/software/apache-flume-1.11.0-bin/lib/

#任务2：配置Flume采集方案

cd /root/software/apache-flume-1.11.0-bin/	## 进入Flume安装目录
mkdir jobs	## 创建目录
cd jobs	## 进入jobs目录
touch logfile_hdfs.conf	## 创建Agent配置文件

vim logfile_hdfs.conf
## 为各组件命名
# taildirSource为a1的Source的名称
a1.sources = taildirSource
# memoryChannel为a1的Channel的名称
a1.channels = memoryChannel
# HDFSSink为a1的Sink的名称
a1.sinks = HDFSSink


# 描述Source
# 数据源Source为TAILDIR类型
a1.sources.taildirSource.type = TAILDIR
# Json格式的文件。记录每个监控文件的inode、最后位置和绝对路径
a1.sources.taildirSource.positionFile = /root/bigdata/data/project2/flumedata/pos_behavior/taildir_position.json
# 以空格分隔的文件组列表。每个文件组指定一组要监测的文件
a1.sources.taildirSource.filegroups = f1
# 被监控文件的绝对路径
a1.sources.taildirSource.filegroups.f1 = /root/bigdata/data/project2/app_log/behavior/.*
# 拦截器，多个用空格分开
a1.sources.taildirSource.interceptors = i1 i2
# 拦截器i1类型，处理标准的JSON格式的数据，如果格式不符合条件，则会过滤掉该数据
a1.sources.taildirSource.interceptors.i1.type = com.qingjiao.flume.interceptor.ETLInterceptor$Builder
# 拦截器i2类型，处理时间漂移的问题，根据日志获取时间把对应的日志文件存放到具体的时间分区目录中
a1.sources.taildirSource.interceptors.i2.type = com.qingjiao.flume.interceptor.TimeStampInterceptor$Builder

## 拼装
# 与Source绑定的Channel
a1.sources.taildirSource.channels = memoryChannel



## 描述Sink
# 接收器Sink为hdfs类型，输出目的地是HDFS
a1.sinks.HDFSSink.type = hdfs
# 数据存放在HDFS上的目录
a1.sinks.HDFSSink.hdfs.path = hdfs://localhost:9000/behavior/origin_log/%Y-%m-%d
# 文件的固定前缀为behavior-
a1.sinks.HDFSSink.hdfs.filePrefix = behavior-
# 时间戳是否需要四舍五入，默认为false，如果为true，则影响除%t之外的所有基于时间的转义序列
a1.sinks.HDFSSink.hdfs.round = false
# 按时间间隔滚动文件，默认30s，此处设置为60
a1.sinks.HDFSSink.hdfs.rollInterval = 60
# 按文件大小滚动文件，默认1024字节，此处设置为134217728字节（128M）
a1.sinks.HDFSSink.hdfs.rollSize = 134217728
# 当Event个数达到该数量时，将临时文件滚动成目标文件，默认是10，0表示文件的滚动与Event数量无关
a1.sinks.HDFSSink.hdfs.rollCount = 0
# 文件格式，默认为SequenceFile，但里面的内容无法直接打开浏览，所以此处设置为DataStream，控制输出文件是原生文件
a1.sinks.HDFSSink.hdfs.fileType = DataStream

## 拼装
# 与Sink绑定的Channel
a1.sinks.HDFSSink.channel = memoryChannel



## 描述Channel
# 缓冲通道Channel为memory内存型
a1.channels.memoryChannel.type = memory
# capacity为最大容量，transactionCapacity为Channel每次提交的Event的最大数量，capacity>= transactionCapacity
a1.channels.memoryChannel.capacity = 1000
a1.channels.memoryChannel.transactionCapacity = 100


## 为各组件命名
# taildirSource为a1的Source的名称
a1.sources = taildirSource
# memoryChannel为a1的Channel的名称
a1.channels = memoryChannel
# HDFSSink为a1的Sink的名称
a1.sinks = HDFSSink

## 描述Source
# 数据源Source为TAILDIR类型
a1.sources.taildirSource.type = TAILDIR
# Json格式的文件。记录每个监控文件的inode、最后位置和绝对路径
a1.sources.taildirSource.positionFile = /root/bigdata/data/project2/flumedata/pos_behavior/taildir_position.json
# 以空格分隔的文件组列表。每个文件组指定一组要监测的文件
a1.sources.taildirSource.filegroups = f1
# 被监控文件的绝对路径
a1.sources.taildirSource.filegroups.f1 = /root/bigdata/data/project2/app_log/behavior/.*
# 拦截器，多个用空格分开
a1.sources.taildirSource.interceptors = i1 i2
# 拦截器i1类型，处理标准的JSON格式的数据，如果格式不符合条件，则会过滤掉该数据
a1.sources.taildirSource.interceptors.i1.type = com.qingjiao.flume.interceptor.ETLInterceptor$Builder
# 拦截器i2类型，处理时间漂移的问题，根据日志获取时间把对应的日志文件存放到具体的时间分区目录中
a1.sources.taildirSource.interceptors.i2.type = com.qingjiao.flume.interceptor.TimeStampInterceptor$Builder

## 描述Sink
# 接收器Sink为hdfs类型，输出目的地是HDFS
a1.sinks.HDFSSink.type = hdfs
# 数据存放在HDFS上的目录
a1.sinks.HDFSSink.hdfs.path = hdfs://localhost:9000/behavior/origin_log/%Y-%m-%d
# 文件的固定前缀为behavior-
a1.sinks.HDFSSink.hdfs.filePrefix = behavior-
# 时间戳是否需要四舍五入，默认为false，如果为true，则影响除%t之外的所有基于时间的转义序列
a1.sinks.HDFSSink.hdfs.round = false
# 按时间间隔滚动文件，默认30s，此处设置为60
a1.sinks.HDFSSink.hdfs.rollInterval = 60
# 按文件大小滚动文件，默认1024字节，此处设置为134217728字节（128M）
a1.sinks.HDFSSink.hdfs.rollSize = 134217728
# 当Event个数达到该数量时，将临时文件滚动成目标文件，默认是10，0表示文件的滚动与Event数量无关
a1.sinks.HDFSSink.hdfs.rollCount = 0
# 文件格式，默认为SequenceFile，但里面的内容无法直接打开浏览，所以此处设置为DataStream，控制输出文件是原生文件
a1.sinks.HDFSSink.hdfs.fileType = DataStream


## 描述Channel
# 缓冲通道Channel为memory内存型
a1.channels.memoryChannel.type = memory
# capacity为最大容量，transactionCapacity为Channel每次提交的Event的最大数量，capacity>= transactionCapacity
a1.channels.memoryChannel.capacity = 1000
a1.channels.memoryChannel.transactionCapacity = 100

## 拼装
# 与Source绑定的Channel
a1.sources.taildirSource.channels = memoryChannel
# 与Sink绑定的Channel
a1.sinks.HDFSSink.channel = memoryChannel




#任务3：使用指定采集方案启动Flume
## 前台启动运行
bin/flume-ng agent -c conf/ -f jobs/logfile_hdfs.conf -n a1 -Dflume.root.logger=INFO,console
# 或者
bin/flume-ng agent --conf conf/ --conf-file jobs/logfile_hdfs.conf --name a1 -Dflume.root.logger=INFO,console
## 后台启动运行（提前创建日志文件所在父目录）
nohup bin/flume-ng agent -c conf/ -f jobs/logfile_hdfs.conf -n a1 -Dflume.root.logger=INFO,console >/root/bigdata/data/project2/flumedata/logs/logfile_hdfs.log 2>&1 &
# 或者
nohup bin/flume-ng agent --conf conf/ --conf-file jobs/logfile_hdfs.conf --name a1 -Dflume.root.logger=INFO,console >/root/bigdata/data/project2/flumedata/logs/logfile_hdfs.log 2>&1 &






