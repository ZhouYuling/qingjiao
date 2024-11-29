package com.qingjiao.analysis.mapper;

import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * 分析数据Mapper继承TableMapper类
 *
 * Mapper的输入key-value类型是：<ImmutableBytesWritable,Result>
 * ImmutableBytesWritable：行键RowKey
 * Result：RowKey对应的一行数据的结果集Result，map()方法每执行一次都接受一个Result结果集
 * 结果集包括<rowkey,family, qualifier, value, timestamp>
 *
 * Mapper的输出key-value类型是:<Text, Text>
 * Text：“主/被叫号码_年份” ||“主/被叫号码_年月份”||“主/被叫号码_年月日”
 * Text： 通话时长duration
 */
public class SumCallDurationMapper extends TableMapper {
    protected void map(ImmutableBytesWritable key, Result value,
                       Mapper<ImmutableBytesWritable, Result, Text, Text>.Context context)
            throws IOException, InterruptedException {
        // 获取行键rowkey，格式为5_19997925219_2023-01-26 14:34:58_19997107000_2550
        String rowkey = Bytes.toString(key.get());

        // 将行键按照分隔符“_”进行切分
        String[] calllogs = rowkey.split("_");
        String call1 = calllogs[1]; // 主叫号码
        String call2 = calllogs[3]; // 被叫号码
        String callTime = calllogs[2]; // 通话建立时间
        String duration = calllogs[4]; // 通话时长
        String year = callTime.substring(0, 4); // 获取年份
        String y_month = callTime.substring(0, 7); // 获取年月份
        String callDate = callTime.substring(0, 10); // 获取年月日

        // 将“主叫号码_年份”作为key，将“通话时长”作为value
        context.write(new Text(call1 + "_" + year), new Text(duration));
        // 将“主叫号码_年月份”作为key，将“通话时长”作为value
        context.write(new Text(call1 + "_" + y_month), new Text(duration));
        // 将“主叫号码_年月日”作为key，将“通话时长”作为value
        context.write(new Text(call1 + "_" + callDate), new Text(duration));

        // 将“被叫号码_年份”作为key，将“通话时长”作为value
        context.write(new Text(call2 + "_" + year), new Text(duration));
        // 将“被叫号码_年月份”作为key，将“通话时长”作为value
        context.write(new Text(call2 + "_" + y_month), new Text(duration));
        // 将“被叫号码_年月日”作为key，将“通话时长”作为value
        context.write(new Text(call2 + "_" + callDate), new Text(duration));
    }
}
