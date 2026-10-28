**服务器已经重新部署，但日志没开。**

在服务器上追加这个开关，再重启：

```bash
printf '\nrequest_log: true\n' >> ~/plain-memory/config.local.yaml
```

```bash
cd ~/plain-memory && kill $(cat run.pid) && sleep 1 && ./start.sh
```


重启后用这条命令验证：

```bash
curl -s http://127.0.0.1:8600/health; echo; tail -n 5 serve.log
```

应该能看到一行 `GET /health 200 …`。

