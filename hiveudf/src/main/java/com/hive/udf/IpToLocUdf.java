package com.hive.udf;

import java.util.List;
import java.nio.charset.StandardCharsets;
import java.util.*;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;

import redis.clients.jedis.Jedis;

/**
 * 自定义UDF函数，实现通过IP获取对应“省份_城市”信息
 *
 * @author root
 *
 */
public class IpToLocUdf extends GenericUDF {
    // 创建一个List列表对象，元素为Map类型
    public static List<Map<String, String>> mapList = new ArrayList<>();
    static {
        String host = "172.17.0.3";// Redis实例所在主机的端口，通过ip addr命令进行查看
        int port = 6379;// Redis实例的端口
        // 1. 获取连接：创建Jedis对象，负责和指定Redis实例进行通信
        Jedis jedis = new Jedis(host, port);
        // 2. 获取“areas”集合中存储的所有元素值,[{"city":"淮安","province":"江苏"}, ...
        // {"city":"南京","province":"江苏"}]
        Set<String> areas = jedis.smembers("areas");
        // 3. 遍历Set集合
        for (String area : areas) {
            // （1）将String类型的Json字符串转化为相应的JSONObject对象
            JSONObject jsonObject = JSON.parseObject(area);
            // （2）创建一个Map集合对象，key和value都为String类型
            Map<String, String> map = new HashMap<>();
            // （3）向Map中添加元素
            map.put("province", jsonObject.getString("province"));
            map.put("city", jsonObject.getString("city"));
            // （4）向List列表中存入Map对象，在不指定位置的情况下插入到List的末尾
            mapList.add(map);
        }
    }

    /**
     * 初始化方法，用来进行参数校验，对象实例化等，返回值类型决定了evaluate方法的返回值类型
     */
    @Override
    public ObjectInspector initialize(ObjectInspector[] objectInspectors) throws UDFArgumentException {
        // 判断传入的参数长度
        if (objectInspectors.length != 1) {
            throw new UDFArgumentLengthException("传入的参数长度不正确!");
        }
        // 判断传入参数的类型，是否为Hive的基本类型，Hive的基本类型都属于PRIMITIVE
        if (!objectInspectors[0].getCategory().equals(ObjectInspector.Category.PRIMITIVE)) {
            throw new UDFArgumentTypeException(0, "传入的参数类型不正确!!!");
        }
        // 返回值类型为Java的String类型
        return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    }

    /**
     * 计算方法，实现具体的业务逻辑，对入参进行转换，此方法最好返回Hadoop标准输出类型，如Text，避免出现强制类型转换异常问题
     *
     * 返回“天津_东丽区”格式的字符串
     *
     */
    @Override
    public Object evaluate(DeferredObject[] deferredObjects) throws HiveException {
        // 判断传入的IP参数是否为NULL
        if (deferredObjects == null || deferredObjects[0] == null || deferredObjects[0].get() == null) {
            return "";
        }
        // 获取下标，Math.random()可以生成[0,1)的double型随机数
        int index = (int) (Math.random() * mapList.size());
        // 使用mapList.get()方法从列表中获取对应下标的元素，元素类型为Map集合，例如：{province=黑龙江, city=哈尔滨}
        // 然后再使用Map.get(key)方法返回指定键所映射的值
        // 格式为：天津_东丽区
        Map<String, String> location = mapList.get(index);
        String pro_city = location.get("province") + "_" + location.get("city");
        // 以UTF-8的格式将String类型转换为Hadoop的标准输出类型Text
        Text new_str = new Text(pro_city);
        return new_str;
    }

    /**
     * 显示UDF的执行计划
     *
     * 类似于Java的toString()方法，Hive中再使用explain命令的时候调用该方法
     */
    @Override
    public String getDisplayString(String[] strings) {
        return ""; // 不打印HQL explain子句中显示的日志，直接返回空字符串
    }
}

