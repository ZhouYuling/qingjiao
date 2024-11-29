#import java.sql.{Connection, DriverManager, PreparedStatement}
#import java.text.SimpleDateFormat
#import java.util.{Calendar, Date, Properties}
#
#import org.apache.kafka.clients.consumer.{ConsumerConfig, ConsumerRecord}
#import org.apache.kafka.common.serialization.StringDeserializer
#import org.apache.spark.streaming.dstream.{DStream, InputDStream}
#import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils, LocationStrategies}
#import org.apache.spark.{SparkConf, SparkContext}
#import org.apache.spark.streaming.{Seconds, StreamingContext}
#
#
#
#/**
#  * 通过Spark Streaming实时处理kafka传递过来的日志数据
#  */
#object Log_Analysis {
#
#  def main(args: Array[String]): Unit = {
#    // TODO 环境准备
#    val conf: SparkConf = new SparkConf().setAppName("kafka-spark_logAnalysis").setMaster("local[*]")
#    val sc: SparkContext = new SparkContext(conf)
#    sc.setLogLevel("WARN")
#    val ssc: StreamingContext = new StreamingContext(sc,Seconds(10))
#
#    // TODO kafka连接参数准备
#    val kafkaParams=Map[String,Object](
#      "bootstrap.servers" ->"???:???",
#      "key.deserializer" -> classOf[StringDeserializer],
#      "value.deserializer" -> classOf[StringDeserializer],
#      "group.id" -> "logAnalysis",
#      //latest:表示如果有offset记录从offset记录开始消费,如果没有从最后/最新的消息开始消费
#      "auto.offset.reset" -> "earliest",
#      "auto.commit.interval.ms" -> "???",//自动提交的时间间隔
#      "enable.auto.commit" -> (true:java.lang.Boolean)//是否自动提交偏移量
#    )
#    // TODO kafka工具类获取数据
#    // 定义主题
#    val topics=Array("logtopic")
#    // kafka主题获取数据
#    val kafkaDS: InputDStream[ConsumerRecord[String, String]] = KafkaUtils.createDirectStream(
#      ssc,
#      LocationStrategies.PreferConsistent,
#      ConsumerStrategies.Subscribe[String, String](topics, kafkaParams)
#    )
#    // 对数据进行切分，获取ip，url
#    val splitDS: DStream[(String, String)] = kafkaDS.map(record => {
#      val value: String = record.value()
#      val splitData: Array[String] = value.split("???")
#        val ip: String = splitData(0)
#        val url: String = splitData(6)
#      (ip, url)
#    })
#    // 数据缓存
#    splitDS.persist()
#
#    // TODO 数据处理分析
#    //(220.248.89.162,/data/attachment/common/cf/200835njz0dz422gdjq44j.png)
#    //splitDS.print()
#    val pv: DStream[Long] = splitDS.map(line=>line._1).count() //访问量
#    val jumper: DStream[Long] = splitDS.map(_._1).map((_,1)).reduceByKey(_+_).filter(_._2==1).count() // 跳转率
#    val reguser: DStream[Long] = splitDS.map(_._2).filter(_.equals("/member.php?mod=register&inajax=1")).count() // 注册用户数
#
#    //pv.print()
#    //jumper.print()
#    //reguser.print()
#    // 获取系统当前时间
#    val time: String = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss").format(new Date())
#    // TODO 数据写入mysql
#    // 遍历访问量pv将值写入mysql
#    pv.foreachRDD(rdd=>rdd.foreachPartition(pvcount=>{
#      // 设置jdbc驱动类
#      Class.forName("com.mysql.jdbc.Driver").newInstance()
#      // 创建数据库连接，设置连接数据库账号密码
#      val conn: Connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/???","???","???")
#      // 编写sql语句插入logpv访问量表数据
#      val sql="insert into logpv(`id`,`pvcount`,`time`) values (null,?,?)"
#      val ps: PreparedStatement = conn.prepareStatement(sql)
#      pvcount.foreach(f=>{
#        // 设置插入值
#        ps.setString(1,f.toString)
#        ps.setString(2,time)
#        ps.executeUpdate()
#      })
#
#    }
#    ))
#    // 遍历跳转率数据并写入mysql
#    jumper.foreachRDD(rdd=>rdd.foreachPartition(jumpercount=>{
#      // 设置jdbc驱动类
#      Class.forName("com.mysql.jdbc.Driver").newInstance()
#      // 创建数据库连接，设置连接数据库账号密码
#      val conn: Connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/hongya","root","123456")
#      // 编写sql语句插入logjumper跳转率表数据
#      val sql="insert into logjumper(`id`,`jumper`,`time`) values (null,?,?)"
#      val ps: PreparedStatement = conn.prepareStatement(sql)
#      jumpercount.foreach(f=>{
#        ps.setString(1,f.toString)
#        ps.setString(2,time)
#        ps.executeUpdate()
#      })
#
#    }))
#    // 遍历注册用户数据并写入mysql
#    reguser.foreachRDD(rdd=>rdd.foreachPartition(regusercount=>{
#      // 设置jdbc驱动类
#      Class.forName("com.mysql.jdbc.Driver").newInstance()
#      // 创建数据库连接，设置连接数据库账号密码
#      val conn: Connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/hongya","root","123456")
#      // 编写sql语句插入logreguser注册表数据
#      val sql="insert into logreguser(`id`,`reguser`,`time`) values (null,?,?)"
#      val ps: PreparedStatement = conn.prepareStatement(sql)
#      regusercount.foreach(f=>{
#        ps.setString(1,f.toString)
#        ps.setString(2,time)
#        ps.executeUpdate()
#      })
#
#    }))
#
#
#    // TODO 启动并等待应用程序优雅退出
#    ssc.start()
#    ssc.awaitTermination()
#    ssc.stop(true,true)
#
#  }
#
#}
