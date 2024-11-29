package com.qingjiao.analysis.reducer;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * 分析数据Reducer继承Reducer类
 *
 * Reducer的输入key-value类型是：<Text,Text>
 * Text：对应Mapper的输出key类型，即“主/被叫号码_年份”||“主/被叫号码_年月份”||“主/被叫号码_年月日”
 * Text：对应Mapper的输出value类型，即通话时长duration
 *
 * Reducer的输出key-value类型是：<Text, Text>
 * Text：不变，仍然是“主/被叫号码_年份”||“主/被叫号码_年月份”||“主/被叫号码_年月日”
 * Text：“通话总次数_通话总时长”
 *
 */
public class SumCallDurationReducer extends Reducer<Text, Text, Text, Text> {
    /*
     * 传入进来的数据格式： <主叫号码_年份,通话时长>
     * <19997924705_2023,0860>
     * <19997924705_2023,0660>
     * <19997925219_2023,2550>
     * ... ...
     *
     * 之后对传入进来的kv对按照key进行分组，此时数据格式为：
     * <19997924705_2023,{0860,0660,...}>
     * <19997925219_2023,{2550,...}>
     * ... ...
     *
     * 每传递一组kv对，调用一次reduce()方法
     */
    @Override
    protected void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context)
            throws IOException, InterruptedException {
        // 定义计数器
        int sumCall = 0; // 用来存储通话总次数
        int sumDuration = 0; // 用来存储通话总时长
        // 遍历一组迭代器，获取通话总次数和通话总时长
        for (Text value : values) {
            int duration = Integer.parseInt(value.toString()); // 获取通话时长
            sumDuration += duration; // 累加通话时长
            sumCall++; // 增加通话次数
        }
        // 将“通话总次数_通话总时长”作为value输出
        String resultValue = sumCall + "_" + sumDuration;
        context.write(key, new Text(resultValue));
    }
}

