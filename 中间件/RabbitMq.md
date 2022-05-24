rabbitmq有啥应用场景呢


spring cloud config 默认使用rabbitmq 发送配置变更通知,依赖spring cloudconfig的组件可以动态更新配置

rabbitmq支持mqtt协议 mqtt协议轻量级  适用于客户端接收告警通知等业务

rabbitmq实时性比kafka好，如果实时性高且数据量不大的情况下，使用rabbitmq更好
