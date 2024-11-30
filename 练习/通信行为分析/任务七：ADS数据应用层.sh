#需求说明
#根据 DWS 层的用户行为数据，我们在 ADS 层再对维度进行上卷，分析维度有：
#
#地域维度：
#需求指标1：统计不同省份用户访问量。
#需求指标2：统计每天不同经济大区用户访问量。
#时间维度：
#需求指标1：统计网站各时间段的用户访问量。
#需求指标2：统计网站各时间段在节假日和工作日时的平均用户访问量。
#网站访问维度：
#需求指标1：不同网站访客的设备类型统计。
#需求指标2：不同网站的上网模式统计。
#需求指标3：不同域名的用户访问量。

#任务一：地域维度
#需求指标1：统计不同省份用户访问量。
create table if not exists ads_user_pro
comment '用户省份分布表'
row format delimited fields terminated by ','
location '/behavior/ads/ads_user_pro'
as select
   province,
   count(*) cnt
from dws_behavior_log
group by province;

#需求指标2：统计每天不同经济大区用户访问量。
create table if not exists ads_user_region
comment '用户经济大区分布表'
row format delimited fields terminated by ','
location '/behavior/ads/ads_user_region'
as select
   t1.dt,
   t2.area,
   count(1) cnt
from dws_behavior_log t1 join dim_area t2
on t1.province=t2.province
group by t1.dt,t2.area;


#任务2：时间维度
select substring(from_utc_timestamp(ts,'Asia/Shanghai'),12,2)
from dws_behavior_log
limit 20;


select t1.visit_hour,count(*) cnt
from
(select substring(from_utc_timestamp(ts,'Asia/Shanghai'),12,2) visit_hour
from dws_behavior_log)t1
group by t1.visit_hour;


create table if not exists ads_user_hour
comment '各时间段用户访问统计表'
row format delimited fields terminated by ','
location '/behavior/ads/ads_user_hour'
as select t1.visit_hour,count(*) cnt
from
(select substring(from_utc_timestamp(ts,'Asia/Shanghai'),12,2) visit_hour
from dws_behavior_log)t1
group by t1.visit_hour;

#需求指标2：统计网站各时间段在节假日和工作日时的平均用户访问量。
select
to_date(from_utc_timestamp(ts,'Asia/Shanghai')) date_id,
substring(from_utc_timestamp(ts,'Asia/Shanghai'),12,2) visit_hour
from dws_behavior_log
limit 20;

select t1.date_id,t1.visit_hour,t2.is_workday,count(*) cnt
from
(select
to_date(from_utc_timestamp(ts,'Asia/Shanghai')) date_id,
substring(from_utc_timestamp(ts,'Asia/Shanghai'),12,2) visit_hour
from dws_behavior_log)t1
join dim_date t2
on t1.date_id=t2.date_id
group by t1.date_id,t1.visit_hour,t2.is_workday;


select t3.visit_hour,t3.is_workday,cast(round(sum(t3.cnt)/count(*)) as int) num
from
(select t1.date_id,t1.visit_hour,t2.is_workday,count(*) cnt
from
(select
to_date(from_utc_timestamp(ts,'Asia/Shanghai')) date_id,
substring(from_utc_timestamp(ts,'Asia/Shanghai'),12,2) visit_hour
from dws_behavior_log)t1
join dim_date t2
on t1.date_id=t2.date_id
group by t1.date_id,t1.visit_hour,t2.is_workday)t3
group by t3.visit_hour,t3.is_workday;


create table if not exists ads_hol_work_user
comment '节假日和工作日各时间段用户访问统计表'
row format delimited fields terminated by ','
location '/behavior/ads/ads_hol_work_user'
as select
t4.visit_hour,
max(case when t4.is_workday=0 then t4.num else null end) as holiday,
max(case when t4.is_workday=1 then t4.num else null end) as workday
from
(select t3.visit_hour,t3.is_workday,cast(round(sum(t3.cnt)/count(*)) as int) num
from
(select t1.date_id,t1.visit_hour,t2.is_workday,count(*) cnt
from
(select
to_date(from_utc_timestamp(ts,'Asia/Shanghai')) date_id,
substring(from_utc_timestamp(ts,'Asia/Shanghai'),12,2) visit_hour
from dws_behavior_log)t1
join dim_date t2
on t1.date_id=t2.date_id
group by t1.date_id,t1.visit_hour,t2.is_workday)t3
group by t3.visit_hour,t3.is_workday)t4
group by t4.visit_hour;

#任务3：网站访问维度
#需求指标1：不同网站访客的设备类型统计。
-- 如果表已存在则先删除
drop table if exists ads_visit_mode;
-- 创建网站访客的设备类型统计表
create external table ads_visit_mode
(
    url         string comment '访问地址',
    device_type string comment '设备类型',
    count       bigint comment '统计数量'
) comment '网站访客的设备类型统计表'
row format delimited fields terminated by '\t'
location '/behavior/ads/ads_visit_mode';


insert overwrite table ads_visit_mode
select url,
       device_type,
       count(1) cnt
from dws_behavior_log
group by url,device_type
order by cnt desc;


#需求指标2：不同网站的上网模式统计。
-- 如果表已存在则先删除
drop table if exists ads_online_type;
-- 创建网站的上网模式统计表
create external table ads_online_type
(
    url   string comment '访问地址',
    type  string comment '上网模式',
    count bigint comment '统计数量'
) comment '网站的上网模式统计表'
row format delimited fields terminated by '\t'
location '/behavior/ads/ads_online_type';


#需求指标3：不同域名的用户访问量。

-- 如果表已存在则先删除
drop table if exists ads_user_domain;
-- 创建域名用户访问统计表
create external table ads_user_domain
(
    domain string comment '访问地址的域名',
    count  bigint comment '统计数量'
) comment '域名用户访问统计表'
row format delimited fields terminated by '\t'
location '/behavior/ads/ads_user_domain';


insert overwrite table ads_user_domain
select t1.domain,
       count(1)
from
(select split(url,'\\.')[1] domain from dws_behavior_log)t1
group by t1.domain;
