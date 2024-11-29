package com.qingjiao.consumer.util;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * 将项目所需的参数外部化
 *
 */
public class PropertiesUtil {
    public static Properties properties = null;// 声明配置对象
    static {
        try {
            // 加载配置属性
            InputStream inputStream = PropertiesUtil.class.getClassLoader().getResourceAsStream("config.properties");
            // 创建配置对象
            properties = new Properties();
            // 从字节输入流中读取属性列表（键和元素对）
            properties.load(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 从属性列表中获取指定键的属性值
     *
     * @param key
     * @return
     */
    public static String getProperty(String key) {
        // 返回指定键的属性值
        return properties.getProperty(key);
    }
}
