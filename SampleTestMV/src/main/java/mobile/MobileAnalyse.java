package mobile;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.math.BigDecimal;

/**
 *
 */
public class MobileAnalyse {

    public static class MobileMapper extends Mapper<LongWritable, Text, Text, MobileBean> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String lines = value.toString();
            if ("0".equals(key.toString())){
                return;
            } else {
                String[] split = lines.split(",");
                String brand = split[0];
                double price = Double.parseDouble(split[2]);
                Integer sales = Integer.valueOf(split[3]);
                Integer rate = Integer.valueOf(split[4]);
                Text k = new Text();
                k.set(brand);
                MobileBean mobileBean = new MobileBean(price, sales, rate);
                context.write(k, mobileBean);
            }
        }

    }

    public static class MobileReducer extends Reducer<Text, MobileBean, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<MobileBean> values, Context context) throws IOException, InterruptedException{
            long total = 0;
            Integer rates = 0;
            Integer sales = 0;
            Integer sum = 0;
            for (MobileBean value : values) {
                sum += 1;
                BigDecimal price = new BigDecimal(value.getPrice());
                BigDecimal sale = new BigDecimal(value.getSales());
                total += price.multiply(sale).longValue();
                sales += value.getSales();
                rates += value.getRate();
            }
            rates = rates / sum;
            Text v = new Text();
            v.set(sales + "\t" + total + "\t" + rates);
            context.write(key, v);
        }
    }

    public static void main(String[] args) throws IOException {

        Configuration conf = new Configuration();
        conf.set("dfs.client.use.datanode.hostname", "true");
        conf.set("fs.defaultFS", "hdfs://hadoop000:8020");
        conf.set("mapreduce.framework.name", "local");

        conf.set("HADOOP_USER_NAME","root");
        System.setProperty("HADOOP_USER_NAME","root"); // 防止提交任务时采用本地系统用户，将本地用户改为远程有权限访问HDFS的用户
        conf.set("mapreduce.job.user.name", "root"); // 指定提交任务的用户


        Job job = Job.getInstance(conf, "IpDataClean");
        job.setJarByClass(MobileAnalyse.class);
        job.setMapperClass(MobileMapper.class);
        job.setReducerClass(MobileReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(MobileBean.class);

        job.setReducerClass(MobileReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        Path inPath = new Path("/mobile.txt");
        Path outPath = new Path("/mobile");
        FileSystem fs = FileSystem.get(conf);

        FileInputFormat.setInputPaths(job, inPath);
        FileOutputFormat.setOutputPath(job, outPath);


        try {
            System.exit(job.waitForCompletion(true) ? 0 : 1);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

    }

}
