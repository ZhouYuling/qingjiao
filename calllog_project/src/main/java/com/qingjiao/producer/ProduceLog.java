package com.qingjiao.producer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Random;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

public class ProduceLog {
    // 存放联系人电话号码与姓名
    public List<String> phoneList = null;

    /**
     * 将文本文件中数据（一行一个字符串）读取到集合中
     *
     * @param filePath
     * @return
     */
    private List<String> readLocalFile(String filePath) {
        BufferedReader reader = null;
        // 创建集合，存放联系人电话与姓名的映射
        phoneList = new ArrayList<String>();
        try {
            // （1）实例化BufferedReader对象reader，使用InputStreamReader作为缓冲输入流的节点流，使用FileInputStream作为InputStreamReader的节点流
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), "UTF-8"));
            /*
             * （2）从数据文件中读取所有的数据，并存入集合
             */
            String line = null;
            // 使用readLine()方法进行整行读取
            while ((line = reader.readLine()) != null) {  // 检查是否还有可读的行
                // 将整行数据添加到phoneList集合中
                phoneList.add(line);
            }

        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            /*
             * （3）释放资源
             */
            if (reader != null) {
                try {
                    reader.close(); // 关闭流
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return phoneList; // 返回存放着联系人电话与姓名的集合
    }


    /**
     * 随机生成通话建立时间
     *
     * @param startDate
     * @param endDate
     * @return
     */
    private String randomDate(String startDate, String endDate) {
        // 将日期字符串按照指定的格式解析为日期对象
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        try {
            // 将开始和结束日期字符串按照指定的格式解析为日期对象
            long startTime = simpleDateFormat.parse(startDate).getTime();
            long endTime = simpleDateFormat.parse(endDate).getTime();
            if (startTime > endTime)
                return null;
            // 通话时间
            Random random = new Random();
            long calltime = startTime + (long) ((endTime - startTime) * random.nextDouble());

            // 通话时间字符串
            return calltime + "";
        } catch (ParseException e) {
            e.printStackTrace();
        }
        return null;
    }


    /**
     * 生产数据
     * 列名	解释	举例
     * call1	主叫号码	19997136006
     * call1_name	主叫人	田立
     * call2	被叫号码	19997139119
     * call2_name	被叫人	汪歌
     * date_time	通话建立时间	2023-07-08 13:55:57
     * duration	通话持续时间（秒）	0574
     *
     * @return
     */
    private String productLog() {
        // 从通讯录中随机查找2个电话号码（主叫，被叫）
        int call1Index = new Random().nextInt(phoneList.size());
        int call2Index = -1;
        String call1 = phoneList.get(call1Index);
        String call2 = null;
        while (true) {
            call2Index = new Random().nextInt(phoneList.size());
            call2 = phoneList.get(call2Index);
            if (call1Index != call2Index) {
                break;
            }
        }
        // 随机生成通话时长(60分钟内)
        int duration = new Random().nextInt(60 * 60); // 60 minutes in seconds;
        // 格式化通话时间，使位数一致（4位数）
        String durationString = new DecimalFormat("0000").format(duration);
        // 生成随机的通话时间，月份：0~11，天：1~31
        String randomDate = randomDate("2023-01-01 00:00:00", "2023-12-31 23:59:59");
        String dateString = randomDate;
        // 拼接log日志
        StringBuilder logBuilder = new StringBuilder();
        logBuilder.append(call1).append("\t").append(call2).append("\t").append(dateString).append("\t")
                .append(durationString);
        System.out.println(logBuilder);
        try {
            Thread.sleep(500);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return logBuilder.toString();
    }


    /**
     * 将产生的日志写入到指定文件中
     *
     * @param filePath
     * @param productLog
     */
    private void writeCallLog(String filePath, ProduceLog productLog) {
        PrintWriter writer = null;
        try {
            // 实例化PrintWriter对象writer，使用OutputStreamWriter作为缓冲输出流的节点流，使用FileOutputStream作为OutputStreamWriter的节点流
            writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(filePath), "UTF-8"));
            while (true) {
                String log = productLog.productLog();
                writer.write(log + "\n");
                writer.flush();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            /*
             * （3）释放资源
             */
            if (writer != null) {
                writer.close();
            }
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("系统参数不正确，请按照指定的格式传递：java -cp producer.jar com.qingjiao.producer.ProduceLog path1 path2");
            System.exit(1);
        }

        ProduceLog produceLog = new ProduceLog();
        // 读取文件数据，获取数据集合
        produceLog.readLocalFile(args[0]);
        produceLog.writeCallLog(args[1], produceLog);
    }


//    -- java -cp 依赖jar或者是依赖jar库 测试类的全限定名 原始文件 目标文件
//java -cp producer.jar com.qingjiao.producer.ProduceLog contact.log calllog.log
}
