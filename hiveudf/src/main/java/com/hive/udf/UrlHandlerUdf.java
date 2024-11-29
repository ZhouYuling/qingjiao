package com.hive.udf;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;

import java.nio.charset.StandardCharsets;

/**
 * 自定义UDF函数，对URL进行处理，实现：
 * （1）将URL中的“http”和“https”协议统一为“http”；
 * （2）截取掉URL后面的查询参数（？后面的参数）。
 *
 * @author root
 *
 */
public class UrlHandlerUdf extends GenericUDF {

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
     * 对URL进行处理，实现：
     * （1）将URL中的“http”和“https”协议统一为“http”；
     * （2）截取掉URL后面的查询参数（？后面的参数）。
     *
     */
    @Override
    public Object evaluate(DeferredObject[] deferredObjects) throws HiveException {
        // 判断传入的URL参数是否为NULL
        if (deferredObjects[0] == null) {
            return "";
        }
        // 获取传入的URL参数，并转换为String类型
        String url = deferredObjects[0].get().toString();
        // 获取“？”在URL中首次出现的位置
        int index = url.indexOf("?");
        if (index > 0) {
            // 截取掉URL？后面的参数
            url = url.substring(0, index);
        }
        // 如果URL以"https://"开头
        if (url.startsWith("https://")) {
            // 将第一次出现的"https://"字符串替换为"http://"
            url = "http://" + url.substring(8);
        }
        // 以UTF-8的格式将String类型转换为Hadoop的标准输出类型Text
        Text new_url = new Text(url.getBytes(StandardCharsets.UTF_8));
        return new_url;
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
