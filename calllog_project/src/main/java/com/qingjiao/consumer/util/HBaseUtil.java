package com.qingjiao.consumer.util;

package com.qingjiao.consumer.util;

import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.NamespaceDescriptor;
import org.apache.hadoop.hbase.NamespaceNotFoundException;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.ColumnFamilyDescriptor;
import org.apache.hadoop.hbase.client.ColumnFamilyDescriptorBuilder;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Table;
import org.apache.hadoop.hbase.client.TableDescriptorBuilder;
import org.apache.hadoop.hbase.util.Bytes;

/**
 * HBase基础数据访问对象
 *
 * 用于封装一些HBase的常用的操作，比如创建命名空间、创建HBase表等
 *
 */
public class HBaseUtil {
    // 定义一个用于放置HBase连接和表管理类的局部线程变量（使每个线程都有自己的连接）
    private static ThreadLocal<Connection> connHolder = new ThreadLocal<Connection>();
    private static ThreadLocal<Admin> adminHolder = new ThreadLocal<Admin>();

    /**
     * 开始获取HBase连接和表管理类
     *
     * @throws Exception
     */
    public static void start() throws Exception {
        getConnection(); // 获取HBase连接对象
        getAdmin(); // 获取表管理类对象
    }

    /**
     * 关闭连接
     *
     * @throws Exception
     */
    public static void end() throws Exception {
        /*
         * 关闭Admin对象的所有资源
         */
        Admin admin = getAdmin();
        if (admin != null) {
            admin.close();
            adminHolder.remove(); // 删除此线程局部变量的当前线程值
        }
        /*
         * 释放HBase数据库连接
         */
        Connection conn = getConnection();
        if (conn != null) {
            conn.close();
            connHolder.remove();// 删除此线程局部变量的当前线程值
        }
    }

    /**
     * 获取HBase连接对象
     */
    public static Connection getConnection() throws Exception {
        Connection connect = connHolder.get();// 从当前线程中获取HBase连接对象
        if (connect == null) {
            // （1）创建HBase配置对象（继承自Hadoop的Configuration，这里使用父类的引用指向子类的对象的设计）
            Configuration config = HBaseConfiguration.create();
            // 通过config.set()方法进行手动设置。设置ZooKeeper队列名称和端口
            config.set("hbase.zookeeper.quorum", "localhost"); // 设置Zookeeper地址
            config.set("hbase.zookeeper.property.clientPort", "2181"); // 设置Zookeeper端口
            // （2）使用连接工厂根据配置器创建与HBase之间的连接对象
            connect = ConnectionFactory.createConnection(config);
            connHolder.set(connect); // 向当前线程中存入HBase连接对象
        }

        return connect; // 返回HBase连接对象
    }

    /**
     * 获取表管理类Admin对象
     */
    public static Admin getAdmin() throws Exception {
        Admin admin = adminHolder.get();// 从当前线程中获取表管理类Admin对象
        if (admin == null) {
            // 通过HBase表连接对象获取表管理类Admin的实例，用来管理HBase数据库的表信息
            admin.close();
            adminHolder.set(admin);// 向当前线程中存入表管理类Admin对象
        }
        return admin; // 返回表管理类Admin对象
    }

    /**
     * 创建命名空间，如果命名空间已经存在，不需要创建；否则，创建新的命名空间
     *
     * @param namespace
     * @throws Exception
     */
    public static void createNamespaceNX(String namespace) throws Exception {
        // （1）获取表管理类Admin的实例，用来管理HBase数据库的表信息
        Admin admin = getAdmin();
        try {
            // （2）获取命名空间（命名空间类似于关系型数据库中的schema，可以想象成文件夹）
            admin.getNamespaceDescriptor(namespace);
        } catch (NamespaceNotFoundException e) {// 如果命名空间不存在，则创建命名空间
            // （3）创建命名空间的描述器
            NamespaceDescriptor ns = NamespaceDescriptor.create(namespace)
                    .addConfiguration("creator", "qingjiao")
                    .addConfiguration("create_time", String.valueOf(System.currentTimeMillis()))
                    .build();
            // （4）使用表管理对象创建命名空间
            admin.createNamespace(ns);
            System.out.println(namespace + " 命名空间创建成功！！！");
        }
    }

    /**
     * 判断HBase表是否存在，如果表已经存在，那么删除后再创建新的HBase表
     *
     * @param tableName
     * @param regionCount
     * @param columnFamilies
     * @throws Exception
     */
    public static void createTableXX(String tableName, Integer regionCount, String... columnFamilies) throws Exception {
        TableName name = TableName.valueOf(tableName);// 表名称
        // 获取表管理类Admin的实例，用来管理HBase数据库的表信息
        Admin admin = getAdmin();
        // 如果表存在则删除，不存在则创建
        if (admin.tableExists(name)) {
            // 删除表
            deleteTable(tableName);
        }
        // 创建表
        createTable(tableName, regionCount, columnFamilies);
    }

    /**
     * 删除HBase表
     *
     * @param tableName
     * @throws Exception
     */
    public static void deleteTable(String tableName) throws Exception {
        TableName name = TableName.valueOf(tableName);// 表名称
        // （1）获取表管理类Admin的实例，用来管理HBase数据库的表信息
        Admin admin = getAdmin();
        // （2）禁用表
		admin.disableTable(name);
        // （3）删除表
		admin.deleteTable(name);
        System.out.println(tableName + " 表删除成功！！！");

    }

    /**
     * 创建HBase表
     *
     * @param tableName
     * @param columnFamilies
     * @throws Exception
     */
    public static void createTable(String tableName, Integer regionCount, String... columnFamilies) throws Exception {
        TableName name = TableName.valueOf(tableName);// 表名称
        // （1）获取表管理类Admin的实例，用来管理HBase数据库的表信息
        Admin admin = getAdmin();
        // （2）创建表描述构建器，定义表的名称
        TableDescriptorBuilder tableDescriptorBuilder = TableDescriptorBuilder.newBuilder(name);
        for (String cf : columnFamilies) {
            // （3）创建列簇描述构建器对象，定义表的列簇
            ColumnFamilyDescriptor family = ColumnFamilyDescriptorBuilder.newBuilder(Bytes.toBytes(cf)).build();
            // （4）通过表描述构建器对象往表中添加列簇
            tableDescriptorBuilder.setColumnFamily(family);
        }
        // （5）使用HBase表管理对象创建普通表及预分区表
        if (regionCount == null || regionCount <= 1) {
            admin.createTable(tableDescriptorBuilder.build()); // 创建普通HBase表
        } else {
            byte[][] splitKeys = getSplitKeys(regionCount); // 获取分区键
            admin.createTable(tableDescriptorBuilder.build(), splitKeys); // 创建预分区HBase表
        }
        System.out.println(tableName + " 表创建成功！！！");
    }

    /**
     * 生成分区键（5个分区键，6个分区）
     *
     * @param regionCount
     * @return
     */
    private static byte[][] getSplitKeys(int regionCount) {
        // 分区键
        int splitKeyCount = regionCount - 1;
        byte[][] bs = new byte[splitKeyCount][];
        // 分区键格式：0|,1|,2|,3|,4|
        // (-∞, 0|), [0|,1|), [1|,2|), [2|,3|), [3|,4|),[4| +∞)
        List<byte[]> bsList = new ArrayList<byte[]>(); // 创建ArrayList集合对象
        // 循环遍历
        for (int i = 0; i < splitKeyCount; i++) {
            String splitkey = i + "|"; // 分区键（i|）
            bsList.add(Bytes.toBytes(splitkey)); // 追加分区键
        }
        bs = bsList.toArray(new byte[bsList.size()][]); // 将ArrayList对象转换成数组，分区键全部存储到byte数组中
        return bs; // 返回存储分区键的数组
    }

    /**
     * 插入数据
     *
     * @param tableName
     * @param put
     * @return
     * @throws Exception
     */
    public static void putData(String tableName, Put put) throws Exception {
        // 获取HBase连接对象，与HBase表进行通信
        Connection connect = getConnection();
        // 创建Table对象，与HBase表进行通信
        Table table = connect.getTable(TableName.valueOf(tableName));
        // 往HBase表中添加数据
        table.put(put);
        // 关闭Table对象的所有资源
        table.close();
    }

    /**
     * 生成分区号(0,1,2,3,4,5)
     *
     * @param call1
     * @param callTime
     * @param regionCount
     * @return
     */
    public static int getPartitionCode(String call1, String callTime, int regionCount) {
        // 取出后4位电话号码，13391501509
        call1.substring(call1.length() - 4);
        // 取出年月，例如202302，2023-02-28 21:10:27
        String yearMonth = callTime.substring(0, 7).replace("-", "");
        // 异或后与初始化设定的region个数求模（crc校验采用异或算法）
        int regionNum = Math.abs(last4Num.hashCode() ^ yearMonth.hashCode()) % regionCount;
        return regionNum; // 返回分区号
    }

    /**
     * 生成行键RowKey
     *
     * @param regionHash
     * @param call1
     * @param calltime
     * @param call2
     * @param duration
     * @return
     */
    public static String getRowKey(int regionNum, String call1, String callTime, String call2, String duration) {
        StringBuilder sb = new StringBuilder(); // 创建可变字符串对象
        // 往对象中追加字符串，格式为regionNum_call1_callTime_call2_duration
        sb.append(regionNum).append("_").append(call1).append("_").append(callTime).append("_").append(call2).append("_").append(duration);
        return sb.toString(); // 将StringBuilder的值转换为String并返回
    }
}

