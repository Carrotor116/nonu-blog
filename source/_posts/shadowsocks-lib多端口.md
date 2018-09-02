---
title: shadowsocks-libev多端口
date: 2018-08-28 03:59:27
tags: linux
id: ss-libev-multi-port
---
## 环境
os: centos 7 
ss: shadowsocks-libev 3.2.0

## 配置方式
### 步骤
1. 创建ss配置文件
2. 创建ss批量启动脚本
3. 配置systemd自启动
<!--more-->

### 创建ss配置文件
新建`/et/shadowsocks-libev/config_x.json`文件
```json
{
    "server":"server_ip",
    "server_port":server_port,
    "local_port":1080,
    "password":"custom password",
    "timeout":300,
    "method":"aes-256-cfb",
    "fast_open":true
}
```

### 创建ss批量启动脚本
新建`/usr/local/bin/shadowsocks-libev-autostart.sh`文件
```shell
#!/bin/sh
proc=/usr/bin/ss-server
config_dir=/etc/shadowsocks-libev
arg=" --fast-open -u "
config_files=()

files=$(ls $config_dir)
for f in $files
do 
  if [[ $f =~ "config_" ]]
  then 
    len=${#config_files[@]}
    config_files[$len]=$f
  fi
done 

for f in ${config_files[@]}
do 
  $proc -c $config_dir/$f $arg -f /var/run/$f.pid
done 
echo "complete. "
```

### 配置systemd自启动
创建文件`/etc/systemd/system/sslibev.service`
```service
[Unit]
Description=Shadowsocks-ssserver
After=network.target

[Service]
Type=forking
TimeoutStartSec=3
ExecStart=/usr/local/bin/shadowsocks-libev-autostart.sh
Restart=always

[Install]
WantedBy=multi-user.target
```
注册systemd启动项，并运行
```bash
 $ sudo systemctl enable /etc/systemd/system/sslibev.service #注册自启动
 $ sudo systemctl start sslibev     # 运行
```

## 使用方式
需要添加新端口的时候，新建`/et/shadowsocks-libev/config_{x}.json`文件，如`config_1.json`、`config_2.json`，配置对应端口和密码，然后重启systemd的`sslibev`服务即可。

## 其他
```bash
 $ sudo systemctl restart sslibev   # 重启sslibev服务
 $ sudo systemctl disable sslibev   # 停用sslibev自启动
 $ sudo rm /etc/systemd/system/sslibev.service   # 移除sslibev启动项
 $ systemctl status sslibev         # 查看运行状态
 $ ss -lnt   # 查看tcp端口接听状态
```

> 参考: [Shadowsocks-libev多端口配置 · 嘘です。が。。。](https://usodesu.ga/2018-06-21/Shadowsocks-libev-Multiple-Port-with-Systemd/)
