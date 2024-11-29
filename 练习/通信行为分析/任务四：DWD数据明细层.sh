

-- 如果表已存在则先删除
drop table if exists dwd_behavior_log;
-- 创建通信行业用户行为日志表
create external table dwd_behavior_log
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
location '/behavior/dwd/dwd_behavior_log'
tblproperties('orc.compress'='snappy');


#任务3：装载数据
set hive.exec.dynamic.partition=true;	-- 开启动态分区功能，默认已开启
set hive.exec.dynamic.partition.mode=nonstrict;  -- 设置为非严格模式

insert overwrite table dwd_behavior_log partition(dt)
select get_json_object(line, '$.client_ip'),
       get_json_object(line, '$.device_type'),
       get_json_object(line, '$.type'),
       get_json_object(line, '$.device'),
       url_trans(get_json_object(line, '$.url')),
       split(get_city_by_ip(get_json_object(line, '$.client_ip')),"_")[0],
       split(get_city_by_ip(get_json_object(line, '$.client_ip')),"_")[1],
       get_json_object(line, '$.time'),
       dt
from ods_behavior_log;


#（1）查看 dwd_behavior_log 表的所有现有分区。
show partitions dwd_behavior_log;
#（2）查看外部表 dwd_behavior_log 的前3行数据。
select * from dwd_behavior_log limit 3;
#（3）统计外部表 dwd_behavior_log 数据总行数。
select count(*) as cnt from dwd_behavior_log;

