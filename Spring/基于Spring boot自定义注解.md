单纯的注解没啥用的，必须要有相应的注解处理器 自定义一些操作

##### 接口统计

定义自定义注解，通过自定义注解定义切点 进行接口统计

```
 @Pointcut("@annotation(com.xue.ta.SystemMetrics)")
    public void controllerAspect() {}
``` 



##### 启动时处理

实现自定义注解

实现自定义注解处理器 比如说通过 `BeanPostProcessor`的一些扩展接口,服务启动时候扫描自定义注解 进行一些初始化操作

比如使用一些分布式任务调度框架,进行提交任务




