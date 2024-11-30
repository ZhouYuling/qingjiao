#任务1：准备驱动
#step1：找到提供的 Hive 连接驱动
cd /root/bigdata/finbi_hive_jars
ls *.gz

#step2：放入 FineBI 的安装目录下
cp *.jar /root/software/FineBI6.0/webapps/webroot/WEB-INF/lib/


#任务2：插件安装
#step1：找到隔离插件
ls /root/bigdata/finbi_hive_jars/fr-plugin-hive-driver-loader-3.0.zip
cd /root/software/FineBI6.0
nohup bin/finebi &

#访问网站：http://IP:37799/webroot/decision
#登录 FineBI 系统，点击「管理系统 -> 插件管理 -> 从本地安装 -> 选择隔离插件」
#重启FineBI
ps -ef | grep finebi
kill -9 3118
nohup bin/finebi &


#任务3：构建Hive连接
#step1：后台启动 HiveServer2 服务
nohup hiveserver2 &
# 或者
nohup hive --service hiveserver2 &

#step2：新建连接
#step3：配置连接
#step4：测试连接：点击「测试连接」，若连接成功则点击「保存」，如下图所示：


#任务4：添加数据库的表至FineBI

