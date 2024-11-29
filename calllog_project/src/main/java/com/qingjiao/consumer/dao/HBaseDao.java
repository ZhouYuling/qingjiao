package com.qingjiao.consumer.dao;

import java.io.IOException;

import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.util.Bytes;

import com.qingjiao.consumer.util.HBaseUtil;
import com.qingjiao.consumer.util.PropertiesUtil;

/**
 * 保存数据
 *
 */
public class HBaseDao {
    private String namespace;// 命名空间
    private String tableName;// HBase表名
    private String cf_caller;// 列簇
    private int regionCount; // 分区数

    /**
     * HBase表初始化
     *
     * @throws Exception
     * @throws IOException
     */
    public void init() throws Exception {
        /*
         * 从属性列表中获取具有指定键的属性值
         */
        namespace = PropertiesUtil.getProperty("hbase.namespace"); // 命名空间
        tableName = PropertiesUtil.getProperty("hbase.table.name"); // HBase表名
        cf_caller = PropertiesUtil.getProperty("hbase.family.name"); // 列簇
        regionCount = Integer.parseInt(PropertiesUtil.getProperty("hbase.regions")); // 分区数
        // 获取HBase连接和表管理类
        HBaseUtil.start();
        // 创建命名空间
        HBaseUtil.createNamespaceNX(namespace);
        // 创建HBase表
        HBaseUtil.createTableXX(tableName, regionCount, cf_caller);
        // 关闭连接
        HBaseUtil.end();
    }

    /**
     * 将通话记录保存到HBase表中
     *
     * @param calllog
     * @throws Exception
     * @throws IOException
     */
    public void insertData(String calllog) throws IOException, Exception {
        // （1）获取通话日志数据，并按照分隔符“\t”进行切分
        String[] log = calllog.split("\t");
        String call1 = log[0]; // 主叫号码
        String call1_name = log[1];// 主叫人姓名
        String call2 = log[2];// 被叫号码
        String call2_name = log[3];// 被叫人姓名
        String callTime = log[4];// 通话建立时间
        String duration = log[5]; // 通话时长
        // （2）获取分区号
        int regionNum = HBaseUtil.getPartitionCode(call1, callTime, regionCount);
        // （3）获取行键RowKey，rowkey = regionNum_all1_callTime_call2_duration
        String rowKey = HBaseUtil.getRowKey(regionNum, call1, callTime, call2, duration);
        // （4）创建Put对象。使用Put对象封装需要添加的信息，一个Put代表一行，构造函数传入的是RowKey
        Put put = new Put(Bytes.toBytes(rowKey));
        // （5）往Put对象上添加信息 （列簇，列，值）
        put.addColumn(Bytes.toBytes(cf_caller), Bytes.toBytes("call1"), Bytes.toBytes(call1));
        put.addColumn(Bytes.toBytes(cf_caller), Bytes.toBytes("call1_name"), Bytes.toBytes(call1_name));
        put.addColumn(Bytes.toBytes(cf_caller), Bytes.toBytes("call2"), Bytes.toBytes(call2));
        put.addColumn(Bytes.toBytes(cf_caller), Bytes.toBytes("call2_name"), Bytes.toBytes(call2_name));
        put.addColumn(Bytes.toBytes(cf_caller), Bytes.toBytes("callTime"), Bytes.toBytes(callTime));
        put.addColumn(Bytes.toBytes(cf_caller), Bytes.toBytes("duration"), Bytes.toBytes(duration));
        // （6）将数据保存到HBase表中
        HBaseUtil.putData(tableName, put);
    }
}
