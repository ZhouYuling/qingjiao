#一、Azkaban安装部署
#1.数据库初始化操作，开启mysql服务，创建azkaban数据
cd /root/software
tar -zxvf azkaban-3.90.0.tar.gz -C /root/software
cd azkaban-3.90.0
./gradew build -x test
systemctl start mysqld
mysql -uroot -p123456
create database azkaban;


#2.给用户qingjiao赋予数据库azkaban所有表的权限
GRANT ALL privileges ON azkaban.* TO 'qingjiao'@'%' WITH GRANT OPTION;


#3.将编译后的azkaban-db安装包解压
mkdir /root/software/azkaban
cd /root/software/azkaban-3.90.0/azkaban-db/build/distributions/
ls
tar azkaban-db-0.1.0-SNAPSHOT.tar.gz -C /root/software/azkaban
find / -name create-all-sql-*.sql


#4.切换至azkaban数据库下，执行上步骤中得到的create-all-sql-*.sql脚本文件进行数据库表初始化
mysql -uroot -p123456
use azkaban;
source /root/software/azkaban/azkaban-db-0.1.0-SNAPSHOT/create-all-sql-0.1.0-SNAPSHOT.sql

#5.将编译后的azkaban-web-server安装包解压
cd /root/software/azkaban-3.90.0/azkaban-web-server/build/distributions/
ls
tar -zxvf azkaban-web-server-0.1.0-SNAPSHOT.tar.gz -C /root/software/azkaban

#6.使用keytool命令生成SSL密钥文件keystore，并将其复移动至解压后的azkaban-web-server-0.1.0-SNAPSHOT目录下
keytool -keystore keystore jetty -genkey -keyalg RSA
#输入123456
#回车
#国家/地区设置为CN
#Y
#口令123456

#7.配置Azkaban Web服务器核心文件，修改默认时区位 亚洲/上海
cd /root/software/azkaban/azkaban-web-server-0.1.0-SNAPSHOT/conf
vim azkaban.properties
default.timezone.id=Asia/Shanghai

#8.默认数据库类型为mysql，对应用户名密码结合前面创建的信息进行更新补充
database.type=mysql
mysql.port=3306
mysql.host=localhost
mysql.database=azkaban
mysql.user=qingjiao
mysql.password=123456
mysql.numconections=100

#9.启动使用SSL连接，端口为8443
jetty.use.ssl=true
jetty.maxThreads=25
jetty.port=8081
jetty.ssl.port=8443

#10.设置jetty对应密码为123456，补充keystore移动后的文件路径
jetty.keystore=/root/software/azkaban/azkaban-web-server-0.1.0-SNAPSHOT/keystore
jetty.password=123456
jetty.keypassword=123456
jetty.truststore=/root/software/azkaban/azkaban-web-server-0.1.0-SNAPSHOT/keystore
jetty.trustpassword=123456

#11.配置用户文件，添加自定义管理用户admin密码admin，对应权限为metrics,admin
vim azkaban.users.xml

<azkaban-users>
<user password="admin" roles="metrics,admin" username="admin">

</azkaban-users>

#12.将编译以后的azkaban-exec-server安装包解压
cd /root/software/azkaban-3.90.0/azkaban-exec-server/build/distributions
ls
tar -zxvf azkaban-exec-server-0.1.0-SNAPSHOT.tar.gz -C /root/software/azkaban

#13.配置azkaban exec核心文件，修改默认时区为 亚洲/上海
cd /root/software/azkaban/azkaban-exec-server-0.1.0-SNAPSHOT/conf
ls
vim azkaban.properties
default.timezone.id=Asia/Shanghai

#14.修改数据库对应用户名和密码
database.type=mysql
mysql.port=3306
mysql.host=localhost
mysql.database=azkaban
mysql.user=qingjiao
mysql.password=123456
mysql.numconections=100

#15.设置executor端口为12321
executor.port=12321

#16解决derby自动载入类问题，将hive安装目录下lib/derby-10.10.2.0.jar复制到Executor Server和Web Server安装到目录lib下
cp /root/software/apache-hive-2.3.4-bin/lib/derby-10.10.2.0.jar /root/software/azkaban/azkaban-exec-server-0.1.0-SNAPSHOT/lib/
cp /root/software/apache-hive-2.3.4-bin/lib/derby-10.10.2.0.jar /root/software/azkaban/azkaban-web-server-0.1.0-SNAPSHOT/lib/

#17.解决log4j版本冲突，将Executor Server和Web Server安装目录lib/slf4j-log4j12-17.21.jar加上.bak后缀
cd /root/software/azkaban/azkaban-exec-server-0.1.0-SNAPSHOT/lib/
mv slf4j-log4j12-17.21.jar slf4j-log4j12-17.21.jar.bak
cd /root/software/azkaban/azkaban-web-server-0.1.0-SNAPSHOT/lib/
mv slf4j-log4j12-17.21.jar slf4j-log4j12-17.21.jar.bak

#18.启动Azkaban Executor Server，并查看日志信息
cd /root/software/azkaban/azkaban-exec-server-0.1.0-SNAPSHOT
bin/start-exec.sh
jps


#19.数据库中激活executors，将active字段的值修改为1
mysql -uroot -p123456
select * from azkaban.executors;
update azkaban.executors set active=1;
select * from azkaban.executors;

#20.启动azkaban web server，并查看日志信息
cd /root/software/azkaban/azkaban-web-server-0.1.0-SNAPSHOT/lib/
bin/start-exec.sh


#二、任务调度管理
#1.添加hadoop 解析记录指向hadoop机器的内网ip地址
hostnamectl set-hostname hadoop000
bash
vim /etc/hosts
内网ip hadoop000

#2.格式化HDFS文件系统
hadoop namenode -format

#3.启动hadoop集群
start-all.sh
#yes

#4.开启mysql服务
systemctl start mysqld

#5.初始化hive元数据，创建hive客户端，创建hive数据库
schematool -dbType mysql -initSchema
hive
create database hive;
