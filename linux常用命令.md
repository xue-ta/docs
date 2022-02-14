- 查看日志 ``` tail -f -n 10 /Log/service*.log```
- 过滤端口抓包 ``` tcpdump -i any -n port 8080 or port 8080 -s0 -w my.pcap -v```
- 查看访问服务的IP ```netstat -anp | grep 8080 | awk '{print $5}' | cut -d ":" -f 1 | uniq -c | sort -nr```
- 进程打开文件数量 ``` lsof -p 544799 | wc -l```
- 查看服务进程 ``` ps -ef ```
- 查看内存 CPU ```top```
- 设置定时任务 ```crontab -e ```
- 压缩日志里面统计关键字 ```zgrep "receive request" ./VCNPCGService_2*.log.zip | wc -l ```

