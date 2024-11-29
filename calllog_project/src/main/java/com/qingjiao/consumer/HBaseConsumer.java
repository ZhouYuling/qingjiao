//package com.qingjiao.consumer;
//
//import java.time.Duration;
//import java.util.Collections;
//
//import org.apache.kafka.clients.consumer.ConsumerRecord;
//import org.apache.kafka.clients.consumer.ConsumerRecords;
//import org.apache.kafka.clients.consumer.KafkaConsumer;
//
//import com.qingjiao.consumer.util.PropertiesUtil;
//
///**
// * 通话日志消费者对象
// *
// * 读取Kafka中缓存的数据，调用HBaseAPI持久化数据
// */
//public class HBaseConsumer {
//    /**
//     * 消费数据
//     *
//     * @throws Exception
//     */
//    public static void main(String[] args) throws Exception {
//        // 使用创建的Properties实例构造消费者KafkaConsumer对象，用于获取Flume采集的数据
//        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(PropertiesUtil.properties);;
//        // 注册要消费的主题（可以消费多个主题）
//        consumer.subscribe(Collections.singletonList(PropertiesUtil.getProperty("kafka.topic")));
//        // 消费数据
//        while (true) {
//            // 阻塞时间，设置10s消费一批数据
//            ConsumerRecords<String, String> consumerRecords = consumer.poll(Duration.ofSeconds(10));
//            // 遍历
//            for (ConsumerRecord<String, String> consumerRecord : consumerRecords) {
//                System.out.println(consumerRecord.value()); // 打印消费到的数据
//            }
//        }
//    }
//}

//（3）启动 Kafka 控制台消费者，等待 Flume 信息的输入：
// kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic calllog --from-beginning

//（4）在 $FLUME_HOME 目录下使用如下命令前台启动 a1：
//bin/flume-ng agent -c conf/ -f jobs/calllog_kafka.conf -n a1 -Dflume.root.logger=INFO,console


//（5）使用如下命令运行通话记录生产脚本“produceLog.sh”：
//sh /root/bigdata/project3/data/produceLog.sh


package com.qingjiao.consumer;

import java.time.Duration;
import java.util.Collections;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;

import com.qingjiao.consumer.dao.HBaseDao;
import com.qingjiao.consumer.util.PropertiesUtil;

/**
 * 通话日志消费者对象
 *
 * 读取Kafka中缓存的数据，调用HBaseAPI持久化数据
 */
public class HBaseConsumer {
    /**
     * 消费数据
     *
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // 使用创建的Properties实例构造消费者KafkaConsumer对象，用于获取Flume采集的数据
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(PropertiesUtil.properties);;
        // 注册要消费的主题（可以消费多个主题）
        consumer.subscribe(Collections.singletonList(PropertiesUtil.getProperty("kafka.topic")));
        // 创建HBase数据访问对象
        HBaseDao hBaseDao = new HBaseDao();
        // HBase表初始化（创建命名空间和HBase表）
        hBaseDao.init();
        // 消费数据
        while (true) {
            // 阻塞时间，设置10s消费一批数据
            ConsumerRecords<String, String> consumerRecords = consumer.poll(Duration.ofSeconds(10));
            // 遍历
            for (ConsumerRecord<String, String> consumerRecord : consumerRecords) {
                System.out.println(consumerRecord.value()); // 打印消费到的数据
                hBaseDao.insertData(consumerRecord.value());// 插入数据
            }
        }
    }
}
