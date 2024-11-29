
#### 1.创建表
# 1. 用户信息表
CREATE TABLE calllog.ct_user (
    id INT(11) NOT NULL AUTO_INCREMENT,
    tel CHAR(11) NOT NULL,
    contact VARCHAR(255),
    PRIMARY KEY (id)
);

# 导入数据
LOAD DATA LOCAL INFILE '/root/bigdata/project3/data/contact.log'
INTO TABLE calllog.ct_user
FIELDS TERMINATED BY ','  -- 假设字段间由逗号分隔
LINES TERMINATED BY '\n'  -- 每行以换行符结束
IGNORE 1 LINES;           -- 忽略第一行，如果文件有标题行的话

# 2. 日期表
CREATE TABLE calllog.call_date (
    id INT(11) NOT NULL AUTO_INCREMENT,
    years CHAR(4) NOT NULL,
    months VARCHAR(2) NULL DEFAULT '',
    days VARCHAR(2) NULL DEFAULT '',
    PRIMARY KEY (id)
);

LOAD DATA LOCAL INFILE '/root/bigdata/project3/data/date_2023.txt'
INTO TABLE calllog.call_date
FIELDS TERMINATED BY ','  -- 假设字段间由逗号分隔
LINES TERMINATED BY '\n'  -- 每行以换行符结束
IGNORE 1 LINES;           -- 如果文件的第一行是标题行，则忽略它


# 3. 通话记录统计表
CREATE TABLE calllog.ct_call (
    id INT(11) NOT NULL AUTO_INCREMENT,
    telid INT(11) NOT NULL,
    dateid INT(11) NOT NULL,
    sumcall INT(11),
    sumduration INT(11),
    PRIMARY KEY (id),
    FOREIGN KEY (telid) REFERENCES calllog.ct_user(id),
    FOREIGN KEY (dateid) REFERENCES calllog.call_date(id)
);




