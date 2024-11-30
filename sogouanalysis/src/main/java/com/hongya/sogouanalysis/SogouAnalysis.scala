package com.hongya.sogouanalysis

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}

object SogouAnalysis {
  // 定义main方法，实现数据读取
  def main(args: Array[String]): Unit = {
    // 创建SparkContext对象
    val sparkConf: SparkConf = new SparkConf()
      .setAppName(this.getClass.getSimpleName.stripSuffix("$"))
      .setMaster("local[*]")
    val sc: SparkContext = new SparkContext(sparkConf)
    sc.setLogLevel("WARN")

    // TODO: 1. 本地读取SogouQ用户查询日志数据
    val rawLogsRDD: RDD[String] = sc.textFile("/root/SogouQ/")
    //println(s"Count = ${rawLogsRDD.count()}")

    // TODO: 2. 解析数据，封装到CaseClass样例类中
    val recordsRDD: RDD[SogouRecord] = rawLogsRDD
      // 过滤不合法数据，如null，分割后长度不等于6
      .filter(log => log != null && log.trim.split("\\s+").length == 6)
      // 对每个分区中数据进行解析，封装到SogouRecord
      .mapPartitions(iter => {
        iter.map(log => {
          val arr: Array[String] = log.trim.split("\\s+")
          SogouRecord(
            arr(0),
            arr(1),
            arr(2).replaceAll("\\[|\\]", ""),//通过正则匹配将“[]"替换成空字符串（HanLp分词无法过滤字符）
            arr(3).toInt,
            arr(4).toInt,
            arr(5)
          )
        })
      })
    // println(s"Count = ${recordsRDD.count()},\nFirst = ${recordsRDD.first()}")

    // 数据使用多次，进行缓存操作，使用count触发
    recordsRDD.persist(StorageLevel.MEMORY_AND_DISK).count()
  }
  /**
   *
   * 用户搜索点击网页记录Record
   * queryTime  访问时间，格式为：HH:mm:ss
   * userId     用户ID
   * queryWords 查询词
   * resultRank 该URL在返回结果中的排名
   * clickRank  用户点击的顺序号
   * clickUrl   用户点击的URL
   *
   */
  case class SogouRecord(
                          queryTime: String,
                          userId: String,
                          queryWords: String,
                          resultRank: Int,
                          clickRank: Int,
                          clickUrl: String
                        )

}
