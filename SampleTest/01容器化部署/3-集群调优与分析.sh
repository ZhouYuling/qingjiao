#1.添加hadoop000解析记录指向hadoop机器的内网ip地址，实现云主机自身使用root访问自身
vim /etc/hosts
ip hadoop000
ssh-keygen
#回车 y 回车
ssh-copy-id hadoop000

#2.格式化hdfs文件系统
hadoop namenode -format

#3.启动hadoop集群
start-all.sh
#yes
jps

#4.故障模拟，杀死namenode进程，删除namenode工作目录数据
#kill -9 namenode_pid

#5.数据恢复，重启新启动namenode进程
stop-all.sh
rm -rf /root/hadoopData/
hadoop namenode -format
start-all.sh

#6.自动进入集群安全模式，文件志接受数据请求

#7.执行命令，退出安全模式
hdfs dfsadmin -safemode get
hdfs dfsadmin -safemode leave

#二、mapreduce分析与优化
#1.将数据/root/data/mobile.txt上传至HDFS根目录下
hadoop dfs -put /root/data/mobile.txt /

#2.根据步骤说明，编写程序，结果保存至HDFS文件系统/mobile目录下

#3.将HDFS上结果文件保存至本地/root/data目录下
hadoop dfs -get /mobile/part-r-00000 /root/data

#三、hive分析与优化
#1.开启mysql服务
systemctl start mysql

#2.初始化hive元数据，进入hive客户端，创建shop数据库
schematool -dbType mysql -initSchema
hive
create database shop;

#3.创建商品表product，并上传本地数据至表内
create external table product (
product_id string,
product_name string,
marque string,
barcode string,
price double,
brand_id string,
market_price double,
stock int,
status int
)
row format delimited fields termminated by ','
;

load data local inpat '/root/data/shop/product.txt' into table product;


#4.创建地区表area，并上传本地数据至表内
create external table area
(area_id string, area_name string) row format delimited fields terminated by ',';

load data local inpat '/root/data/shop/area.txt' into table area;

#5.创建用户点击usesr_click，并上传本地数据至表内
create external table user_click
(user_id string,user_ip string,url string,click_time string,action_type string)
row format delimited fields terminated by ',';

load data local inpat '/root/data/shop/user_click.txt' into table user_click;

#6.创建用户点击商品日志表clicklog，解析user_click用户点击信息表中的product_id
create table external table clicklog
(user_id string,user_ip string,product_id string,click_time string,action_type string,area_id string)
row format delimited fields terminated by ',';

insert into table clicklog
select user_id,user_ip,substring(url,instr(url,"=")+1),
click_time,action_type,area_id from user_click;

#7.创建结果分析区域热门area_hot_product，统计各地区热门商品访问量pv
create table area_hot_product
(area_id string,area_name string,product_id string,product_name string,pv bigint)
row format delimited fields terminated by ',';

insert into table area_hot_product
select a.area_id,b.area_name,a.product_id,c.product_name,count(a.product_id) pv
from clicklog a join area b on a.area_id=b.area_id join product c on a.product_id = c.product_id
group by a.area_id,b.area_name,a.product_id,c.product_name
;

#8.查询表area_hot_product全部数据，结果写入本地目录/root/data/shop/area_hot_product
insert overwrite local directory '/root/data/shop/area_hot_product'
row format delimited fields terminated by '\t'
select * from area_hot_product;
