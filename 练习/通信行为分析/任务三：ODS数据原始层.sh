#任务1：创建数据库
create database if not exists comm;	# 创建数据库
use comm;	# 切换数据库

#任务2：创建ODS层数据表
-- 如果表已存在则先删除
drop table if exists ods_behavior_log;
-- 创建通信行业用户行为日志源表
create external table ods_behavior_log
(
    line string
) comment '通信行业用户行为日志源表'
partitioned by (dt string)
location '/behavior/ods/ods_behavior_log';

#任务3：加载数据
load data inpath '/behavior/origin_log/2023-01-01'
into table ods_behavior_log partition (dt='2023-01-01');
load data inpath '/behavior/origin_log/2023-01-02'
into table ods_behavior_log partition (dt='2023-01-02');
load data inpath '/behavior/origin_log/2023-01-03'
into table ods_behavior_log partition (dt='2023-01-03');
load data inpath '/behavior/origin_log/2023-01-04'
into table ods_behavior_log partition (dt='2023-01-04');
load data inpath '/behavior/origin_log/2023-01-05'
into table ods_behavior_log partition (dt='2023-01-05');
load data inpath '/behavior/origin_log/2023-01-06'
into table ods_behavior_log partition (dt='2023-01-06');
load data inpath '/behavior/origin_log/2023-01-07'
into table ods_behavior_log partition (dt='2023-01-07');

show partitions ods_behavior_log;

