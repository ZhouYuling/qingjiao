# 任务1：创建DWS层数据表
-- 如果表已存在则先删除
drop table if exists dws_behavior_log;
-- 创建通信行业用户行为日志表
create external table dws_behavior_log
(
    client_ip   string comment '客户端IP',
    device_type string comment '设备类型',
    type        string comment '上网模式',
    device      string comment '设备ID',
    url         string comment '访问的资源路径',
    province    string comment '省份',
    city        string comment '城市',
    ts          bigint comment '时间戳'
) comment '通信行业用户行为日志表'
partitioned by(dt string)
stored as orc
location '/behavior/dws/dws_behavior_log'
tblproperties('orc.compress'='snappy');

#任务2：装载数据
set hive.exec.dynamic.partition=true;	-- 开启动态分区功能，默认已开启
set hive.exec.dynamic.partition.mode=nonstrict;  -- 设置为非严格模式

insert overwrite table dws_behavior_log partition(dt)
select client_ip,
       device_type,
       type,
       device,
       url,
       province,
       city,
       ts,
       dt
from dwd_behavior_log;

show partitions dws_behavior_log;

select count(*) as cnt from dws_behavior_log;


