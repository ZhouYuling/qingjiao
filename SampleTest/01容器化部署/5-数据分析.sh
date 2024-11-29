


mkdir -p /root/chat/data

#敲hive进入hive客户端
create database if not exists msg;
use msg;

create external table msg.ods_chat(
msg_time string comment '消息发送时间',
sender_name string comment '发送人昵称',
sender_account string comment '发送人账号',
sender_gender string comment '发送人性别',
sender_ip string comment '发送人IP地址',
sender_os string comment '发送人操作系统',
sender_phonemodel string comment '发送人手机型号',
sender_network string comment '发送人网络类型',
sender_gps string comment '发送人的GPS定位',
receiver_name string comment '接收人昵称',
receiver_ip string comment '接收人IP地址',
receiver_account string comment '接收人账号',
receiver_os string comment '接收人操作系统',
receiver_phonetype string comment '接收人手机型号',
receiver_network string comment '接收人网络类型',
receiver_gps string comment '接收人的GPS定位',
receiver_gender string comment '接收人性别',
msg_type string comment '消息类型',
distance string comment '双方距离',
message string comment '消息内容')
row format delimited fields terminated by '\t';


load data local inpath '/root/chat/data/*'
into table msg.ods_chat;

select * from ods_chat limit 3;
select msg_time,sender_name,sender_gps
from msg.ods_chat
where length(sender_gps)=0
limit 10;

select * from msg.ods_chat
where length(sender_gps)>0;

select msg_time from msg.ods_chat limit 10;

select
msg_time,
to_date(msg_time) as dayinfo, --获取天
substr(msg_time,12,2) as hourinfo --获取小时
from msg.ods_chat
limit 5;

# 123.257181,48.807394
select
sender_gps,
split(sender_gps,',')[0] as sender_Ing, --提取经度
split(sender_gps,',')[1] as sender_lat --提取纬度
from msg.ods_chat



create external table if not exists msg dwd_chat_etl(
msg_time string comment '消息发送时间',
sender_name string comment '发送人昵称',
sender_account string comment '发送人账号',
sender_gender string comment '发送人性别',
sender_ip string comment '发送人IP地址',
sender_os string comment '发送人操作系统'
sender_phonemodel string comment '发送人手机型号',
sender_network string comment '发送人网络类型',
sender_gps string comment '发送人的GPS定位',
sender_lng string comment '发送人的GPS经度',
sender_lat string comment '发送人的GPS纬度',
receiver_name string comment '接收人昵称',
receiver_ip string comment '接收人IP地址O',
receiver_account string comment '接收人账号',
receiver_os string comment'接收人操作系统',
receiver_phonetype string comment '接收人手机型号',
receiver_network string comment '接收人网络类型',
receiver_gps string comment '接收人的GPS定位',
receiver_gender string comment '接收入性别',
msg_type string comment '消息类型',
distance string comment '双方距离',
message string comment '消息内容')
comment'移动社交行业聊天记录分区表
partitioned by(dayinfo string,hourinfo string)
stored as orc;


set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;

insert overwrite table msg.dwd_chat_etl partition(dayinfo,hourinfo)
select msg_time, sender_name, sender_account, sender_gender,
sender_ip, sender_os, sender_phonemodel, sender_network, sender_gps,
split(sender_gps,',')[0] as sender_Ing, --提取经度
split(sender_gps,',')[1] as sender_lat, --提取纬度
receiver_name, receiver_ip, receiver_account, receiver_os, receiver_phonetype, receiver_network,
receiver_gps, receiver_gender,msg_type,distance,message,
to_date(msg_time) as dayinfo, --获取天
substr(msg_time,12,2) as hourinfo --获取小时
from msg.ods_chat
where length(sender_gps)>0;

show partitions msg.dwd_chat_etl;


create external table if not exists msg dws_chat_etl(
msg_time string comment '消息发送时间',
sender_name string comment '发送人昵称',
sender_account string comment '发送人账号',
sender_gender string comment '发送人性别',
sender_ip string comment '发送人IP地址',
sender_os string comment '发送人操作系统'
sender_phonemodel string comment '发送人手机型号',
sender_network string comment '发送人网络类型',
sender_gps string comment '发送人的GPS定位',
sender_lng string comment '发送人的GPS经度',
sender_lat string comment '发送人的GPS纬度',
receiver_name string comment '接收人昵称',
receiver_ip string comment '接收人IP地址O',
receiver_account string comment '接收人账号',
receiver_os string comment'接收人操作系统',
receiver_phonetype string comment '接收人手机型号',
receiver_network string comment '接收人网络类型',
receiver_gps string comment '接收人的GPS定位',
receiver_gender string comment '接收入性别',
msg_type string comment '消息类型',
distance string comment '双方距离',
message string comment '消息内容')
comment'移动社交行业聊天记录分区表
partitioned by(dayinfo string,hourinfo string)
stored as orc
tblproperties('orc.compress'='snappy')
;

set hive.exec.dynamic.partition=true; -- 开启动态分区功能，默认已开启
set hive.exec.dynamic.partition.mode=nonstrict; -- 设置为非严格模式
insert overwrite table msg.dws_chat partition(dayinfo,hourinfo)
select * from msg.dwd_chat_etl;

#3.数据分析要求如下：
#需求指标1：统计单日消息量
#需求指标2：统计单日内不同时段消息量分布
#需求指标3：统计单日不同时段下不同性别发送消息数
#需求指标4：统计单日发送消息最多的Top10用户
#需求指标5：统计单日接收消息最多的Top10用户
#需求指标6：查找关系最亲密的10对好友
#需求指标7：统计单日各地区发送消息数据量。

#1.需求指标1：统计单日消息量，结果到出至本地/msg/ads/hour_msg_cn路径下
#分析单日消息总量，为平台运维团队提供平台弹性扩容支持决策，实现资源优化配置，节约成本

insert overwrite local directory '/msg/ads/hour_msg_cn'
row format delimited fields terminated by ',
select
count(*) as day_cnt
from msg.dws_chat
group by dayinfo;

#2.需求指标2：统计单日内不同时段消息量分布，将统计结果导出到本地/msg/ads/hour_msg_cnt目录中，并指定字段的分隔符为，
#数据参考：00,4349，为00:00-00:59期间消息总量
#用户在上网时间上有一定规律，将分析粒度按小时划分为24个区间，编号为00-23，
#对应一天的24个时间段(即00:00-00:59:59为00)。
#根据消息量时间分布统计，分析不同时段里用户的活跃度情况，为运营部门提供广
#告溢价决策支持，为广告厂商提供广告投放决策支持。
#操作环境：Hadoop- hive

insert overwrite local directory '/msg/ads/hour_msg_cnt'
row format delimited fields terminated by ','
select
dayinfo,
hourinfo,
count(*) as hour_cnt
from msg.dws_chat
group by dayinfo,hourinfo;


#3.需求指标3：统计单日不同时段下不同性别发送消息数，将统计结果导出到本地的/msg/ads/hour_gender_cnt目录中，并指定字段的分隔符为“,”
#结果输出如下格式：
#dayinfo time_span female male
#2022-11-01 上午 4132 11357
#（1）凌晨的时间段为：01:00:00~04:59:59
#（2）早上的时间段为：05:00:00~07:59:59
#(3）上午的时间段为：08:00:00~10:59:59
#(4）中午的时间段为：11:00:00~12:59:59
#（5）下午的时间段为：13:00:00~16:59:59
#（6）傍晚的时间段为：17:00:00~18:59:59
#（7）晚上的时间段为：19:00:00~22:59:59
#（8）子夜的时间段为：23:00:00~00:59:59

insert overwrite local directory '/msg/ads/hour_gender_cnt'
row format delimited fields terminated by ','
select
b.dayinfo,b.time_span,
max(case when b.sender_gender = '女' then b.cnt else null end) as female,
max(case when b.sender_gender = '男' then b.cnt else null end) as male
from
(

select
a.dayinfo,
a.time_span,
a.sender_gender,
count(*) as cnt
from (

select hourinfo,
case
when hourinfo < 1 or hourinfo >= 23 then '子夜'
when hourinfo < 5 then '凌晨'
when hourinfo < 8 then '早上'
when hourinfo < 11 then '上午'
when hourinfo < 13 then '中午'
when hourinfo < 17 then '下午'
when hourinfo < 19 then'傍晚'
when hourinfo < 23 then '晚上'
end as time_span
from msg.dws_chat

) as a
group by a.dayinfo, a.time_span, a.sender_gender

) as b
group by b.dayinfo,b.time_span
;


#4.需求指标4：統计单日发送消息最多的Top10用户
#数据参考： 始鸾，768，为用户发送消息总量
#将统计结果导出到本地的/msg/ads/susr_top10目录中

insert overwrite local directory '/msg/ads/susr_top10'
row format delimited fields terminated by ',
select
dayinfo,
sender_name as username,
count(*) as sender_msg_cnt
from msg.dws_chat
group by dayinfo,sender_name
order by sender_msg_cnt desc
limit 10;

#5.需求指标5：统计单日接收消息最多的Top10用户
#数据参考： 始鸾，768，为用户接受消息总量
#将统计结果导出到本地的/msg/ads/rusr-top1e目录中,并指定字段的分隔符为
#统计发送和接收消息最多的Top10用户,有助于产品定位目标人群,构造用户画像,这
#是用户运营工作的第一步。用户画像可以围绕产品进行人群细分，确定产品的核心一
#人群，从而有助于确定产品定位,优化产品的功能点。

insert overwrite local directory '/msg/ads/rusr_top10'
row format delimited fields terminated by ','
select
dayinfo,
receiver_name as username,
count(*) as receiver_msg_cnt
from msg.dws_chat
group by dayinfo,receiver_name
order by receiver_msg_cnt desc
limit 10;

#6.需求指标6：查找关系最亲密的10对好友。
#数据参考： 张三、李四，19 ，一对好友之间的联系次数
#根据聊天记录中** sender_name（发送人昵称）**和 receiver_name(接收人昵称），统计出最经常联系的10对好友。要求将统计结果导出到本地的
#/msg/ads/chat_friend 目录中，并指定字段的分隔符为“.”。
#亲密关系对于用户持续活跃、提升新用户获取速度和获取能力、增强付费能力等
#都带来了非常大帮助。产品可以研发亲密度通讯录，统计并管理每个联系人的亲密。
#关系，利用联系人亲密度来搜索和排序，提高沟通效率。

insert overwrite local directory '/msg/ads/chat_friend'
row format delimited fields terminated by ',
select
case when sender_name <= receiver_name then sender_name else receiver_name end as user1,
case when sender_name > receiver_name then sender_name else receiver_name end as user2,
count(*) as frequency
from msg.dws_chat
group by
case when sender_name <= receiver_name then sender_name else receiver_name end,
case when sender_name > receiver_name then sender_name else receiver_name end
order by frequency desc
limit 10;


#7.需求指标7：.统计单日各地区发送消息数据量。
#数据参考： 202-11-01 100.297355,24.206808 100.297355 24.206808 723，结合经纬度其地址可查找为云南省临沧市，发送消息量为723次。
#将统计结果（要求包含 dayinfo 、 sender_gps、 longitude 和 latitude 字段）导出到本地的 /msg/ads/loc_msg_cnt 目录中，并指定字段的分隔符为“\t”。
#分析用户地域分布规律，为不同区域的市场推广策略提供支撑数据支撑，为平台数据在全国的分布式节点选择提供数据支持。

insert overwrite local directory '/msg/ads/loc_msg_cnt'
row format delimited fields terminated by '\t'
select
dayinfo,
sender_gps,
cast(sender_lng as double) as longitude,
cast(sender_lat as double) as latitude,
count(*) as loc_cnt
from msg.dws_chat
group by dayinfo,sender_gps,sender_Ing,sender_lat;

