

#
vim core-site.xml

<configuration>
<property>
<name>hadoop.tmp.dir</name>
<value>/D:/java/hadoop-2.7.7/files/tmp</value>
</property>
<property>
<name>dfs.name.dir</name>
<value>/D:/java/hadoop-2.7.7/files/name</value>
</property>
<property>
<name>fs.default.name</name>
<value>hdfs://localhost:9000</value>
</property>
</configuration>


vim hdfs-site.xml

<configuration>

<property>
<name>dfs.replication</name>
<value>1</value>
</property>
<property>
<name>dfs.data.dir</name>
<value>/D:/java/hadoop-2.7.7/files/data</value>
</property>
</configuration>

vim mapred-site.xml

<configuration>
<property>
<name>mapreduce.framework.name</name>
<value>yarn</value>
</property>
<property>
<name>mapred.job.tracker</name>
<value>hdfs://localhost:9001</value>
</property>
</configuration>

vim yarn-site.xml

<configuration>
<property>
<name>yarn.nodemanager.aux-services</name>
<value>mapreduce_shuffle</value>
</property>
<property>
<name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
<value>org.apache.hadoop.mapred.ShuffleHandler</value>
</property>
</configuration>