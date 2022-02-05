xconfig在spring cloud config基础上进行了定制化开发，配置项存储在pg数据库中，当xconfig上的配置项发生变更时，用
@RefreshScope @ConfigurationProperties 注解的bean会实时动态更新


## 服务端


1. server发布了RefreshRemoteApplicationEvent事件 
applicationContext.publishEvent(new RefreshRemoteApplicationEvent(this, applicationContext.getId(), null));
   
2. server端spring cloud bus(org.springframework.cloud.bus.org.springframework.cloud.bus#acceptLocal)监听 RemoteApplicationEvent，并通过rabbitmq发布给 client
    @EventListener(classes = RemoteApplicationEvent.class)
    public void acceptLocal(RemoteApplicationEvent event) {
        if (this.serviceMatcher.isFromSelf(event)
                && !(event instanceof AckRemoteApplicationEvent)) {
            this.cloudBusOutboundChannel.send(MessageBuilder.withPayload(event).build());
        }
    }


## 客户端


1.spring cloud bus (org.springframework.cloud.bus.org.springframework.cloud.bus#acceptRemote) 监听总线消息
    @StreamListener(SpringCloudBusClient.INPUT)
    public void acceptRemote(RemoteApplicationEvent event)()
    
2. 发布 RefreshRemoteApplicationEvent
this.applicationEventPublisher.publishEvent(event)


3.org.springframework.cloud.bus.event#RefreshListener  处理配置刷新事件


3.1 
    public synchronized Set<String> refresh() {
        Map<String, Object> before = extract(
                this.context.getEnvironment().getPropertySources());
        //重新构建一个 springApplicationContext
        addConfigFilesToEnvironment();
        Set<String> keys = changes(before,
                extract(this.context.getEnvironment().getPropertySources())).keySet();
        //发布配置变更事件，@ConfigurationProperties 注解的bean会动态更新
        this.context.publishEvent(new EnvironmentChangeEvent(context, keys));
        //@refreshScope注解的bean更新
        this.scope.refreshAll();
        return keys;
    }


3.2 refresh->addConfigFilesToEnvironment   重新构建一个springApplication


            StandardEnvironment environment = copyEnvironment(
                    this.context.getEnvironment());
            SpringApplicationBuilder builder = new SpringApplicationBuilder(Empty.class)
                    .bannerMode(Mode.OFF).web(false).environment(environment);
            builder.application()
                    .setListeners(Arrays.asList(new BootstrapApplicationListener(),
                            new ConfigFileApplicationListener()));
            capture = builder.run();

3.3 @refreshScope注解的bean更新
org.springframework.cloud.context.scope.refresh.RefreshScope refreshAll
Dispose of the current instance of all beans in this scope and force a refresh on next method execution
销毁此范围内所有bean的当前实例
只有在下一次getBean的时候才会重建@refreshScope 注解的bean
3.4 ConfigurationPropertiesRebinder  监听EnvironmentChangeEvent事件
                this.applicationContext.getAutowireCapableBeanFactory().destroyBean(bean);
                this.applicationContext.getAutowireCapableBeanFactory().initializeBean(bean, name);


4.需要注意的是，当远程server中配置项更新的时候，只是发布了refreshApplicationEvent，通知client有配置项变更，具体的变更需要client调用server接口重新获取


4.1 注意configServicePropertySource#locate作用就是clienthttp请求从sever上获取配置，放入到spring 容器
4.2 locate方法是在什么时机被调用呢？
查看bean PropertySourceBootstrapConfiguration，继承了ApplicationContextInitializer，在spring boot 进行初始化的时候调用，将所有PropertySourceLocator类型的对象的locate方法循环调用，然后将各个locate获取的属性值放到
composite中利用insertPropertySources(propertySources, composite)设置到environment中