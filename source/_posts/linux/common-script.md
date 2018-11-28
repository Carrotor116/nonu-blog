---
title: 常用脚本
id: common-script
date: 2018-09-30 10:43:31
tags: script
---
android 文件共享脚本

<!-- more -->



---



## http.server 
android 中使用 termux 开启文件共享web.

需要安装`python 3`，将脚本移动至`/data/data/com.termux/files/usr/bin`目录下使用。



> Usage: scriptName [port]
> &nbsp;&nbsp;&nbsp;&nbsp; port &nbsp;&nbsp;&nbsp;&nbsp;  web端口号，默认使用 2000



```bash 
#!/data/data/com.termux/files/usr/bin/bash

port=2000

if [ "$1" -gt 0 ] 2>/dev/null ;then
    port=$1
fi

echo "http.server run on port:" $port

ip -c -4 add | awk '/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}..[0-9]{1,3}/{print $2}'

python -m http.server $port
```
