

#任务1：创建维度表
-- 如果表已存在则先删除
drop table if exists dim_date;
-- 创建时间维度表
create external table dim_date
(
    date_id string comment  '日期',
    week_id string comment  '周',
    week_day string  comment  '星期',
    day string comment '一个月的第几天',
    month string comment '第几个月',
    quarter string comment '第几个季度',
    year string comment '年度',
    is_workday string comment '是否是工作日',
    holiday_id string comment '国家法定假日'
) comment '时间维度表'
row format delimited fields terminated by '\t'
location '/behavior/dim/dim_date'
tblproperties('skip.header.line.count'='1');	-- 忽略表头，过滤首行


#2. 创建地区维度表
-- 如果表已存在则先删除
drop table if exists dim_area;
-- 创建地区维度表
create external table dim_area
(
    city string comment '城市',
    province string comment '省份',
    area string comment '地区'
) comment '地区维度表'
row format delimited fields terminated by '\t'
location '/behavior/dim/dim_area';


#任务2：加载数据
load data local inpath '/root/bigdata/data/project2/app_log/dimension/dim_date_2023.txt'
into table dim_date;
load data local inpath '/root/bigdata/data/project2/app_log/dimension/dim_area.txt'
into table dim_area;


select * from dim_date limit 3;
select * from dim_area limit 3;


select count(*) as cnt from dim_date;
select count(*) as cnt from dim_area;



