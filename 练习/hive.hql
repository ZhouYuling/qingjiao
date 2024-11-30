systemctl start mysqld

schematool -dbType mysql -initSchema
hive
create database shop

create table product(
    product_id string,
    product_name string,
    marque string,
    barcode string,
    price double,
    brand_id string,
    market_price double,
    stock int,
    status int
) row format delimited fields terminated by ',';
load data local inpath "/root/data/shop/product.txt" into table product;

create table area(
    area_id string,
    area_name string
)row format delimited fields terminated by ',';

load data local inpath "/root/data/shop/area.txt" into table area;

create table user_click(
    user_id string,
    user_ip string,
    url string,
    click_time string,
    action_type string,
    area_id string
)row format delimited fields terminated by ',';

load data local inpath "/root/data/shop/user_click.txt" into table user_click;

create table clicklog(
    user_id string,
    user_ip string,
    product_id string,
    click_time string,
    action_type string,
    area_id string
)row format delimited fields terminated by ',';

insert into table clicklog
select user_id, user_ip, substring(url, instr(url, "=") + 1),click_time,action_type,area_id from user_click;

create table area_hot_product(
    area_id string,
    area_name string,
    product_id string,
    product_name string,
    pv bigint
)row format delimited fields terminated by ',';

insert into table area_hot_product
select a.area_id, b.area_name, a.product_id, c.product_name, count(a.product_id) pv
from clicklog a join area b on a.area_id = b.area_id join product c on a.product_id = c.product_id 
group by a.area_id, b.area_name, a.product_id, c.product_name;

insert overwrite local directory '/root/data/shop/area_hot_product' select * from area_hot_product;