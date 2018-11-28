---
title: "shell awk"
id: awk
tags: ['shell']
date: 2018-02-19
---

awk 以输入文件的一行为处理单位，进行匹配和分析

<!--more-->



## awk

```shell
$ awk '{pattern + action}' {filename}
```
pattern使用`/`包围，如`/root/`匹配含有root的行，可以使用正则
action使用`{}`包围。
如`awk -F: '/^root/ {print $1 ", " $ 7}' /etc/passwd`，匹配`root`开头的行，输出第一个域和第七个域。



## 栗子1
命令
`awk -F ':' 'BEGIN {print "name, shell"} {print $1", "$7} END {print "blue, /bin/nosh"}'  /etc/passwd`
每个`action`使用`{}`包围，一个`{}`内可以有多个action，需要使用`;`分割，如`'{count++; print $0} END {print "count is ", count}'`，

结果:
```shell
$ awk -F ':' 'BEGIN {print "name, shell"} {print $1", "$7} END {print "blue, /bin/nosh"}'  /etc/passwd
name, shell
root, /bin/bash
daemon, /usr/sbin/nologin
    ....
nonu, /bin/bash
blue, /bin/nosh
```



## 栗子2
命令`$ awk -F: '/root/' /etc/passwd`
匹配`root`的域，没有指定action，默认`print $0`打印全部
匹配可以使用正则表达式

结果:
```shell
$ awk -F: '/root/' /etc/passwd
root:x:0:0:root:/root:/bin/bash
```



## 栗子3
命令`$ awk -F: '/root/ {print $1 ", " $ 7}' /etc/passwd`
匹配是按整行的

结果:
```shell
$ awk -F: '/root/ {print $1 ", " $ 7}' /etc/passwd
root, /bin/bash
```



## 栗子4

`FILENAME`, `NR`, `NF`, `$0` 的含义

```shell
$ awk -F: '{printf("filename:%10s, linenumber:%s, columns:%s, linecontent:%s\n", FILENAME, NR, NF, $0)}' /etc/passwd
filename:/etc/passwd, linenumber:1, columns:7, linecontent:root:x:0:0:root:/root:/bin/bash
filename:/etc/passwd, linenumber:2, columns:7, linecontent:daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
    .....
filename:/etc/passwd, linenumber:30, columns:7, linecontent:pollinate:x:111:1::/var/cache/pollinate:/bin/false
filename:/etc/passwd, linenumber:31, columns:7, linecontent:nonu:x:1000:1000:,,,:/home/nonu:/bin/bash
```



## 栗子5
统计文件大小，含文件夹（linux 下文件夹大小为 4096）

```shell
$ ll | awk 'BEGIN{size=0} {size=size+$5} END{print "size=" size/1024/1024 "M"}'
size=0.0257177M
```


统计文件大小，不含文件夹
```shell
$ ll | awk 'BEGIN{size=0} {if ($5 != 4096) {size+=$5}} END{print "size = " size/1024/1024 "M"}'
size = 0.0257177M
```

```shell
$ ll
total 32
drwxr-xr-x 0 nonu nonu   512 Feb 19 13:16 .
drwxr-xr-x 0 root root   512 Jan  7 20:51 ..
-rw------- 1 nonu nonu 15395 Feb 18 22:51 .bash_history
-rw-r--r-- 1 nonu nonu   220 Jan  7 20:51 .bash_logout
-rw-r--r-- 1 nonu nonu  3837 Feb 19 13:16 .bashrc
drwxrwxrwx 0 nonu nonu   512 Jan  7 20:53 info
-rwxrwxrwx 1 nonu nonu    45 Jan 18 20:12 moshb
-rwxrwxrwx 1 nonu nonu    44 Jan  7 20:38 moshcloudcentos
-rwxrwxrwx 1 nonu nonu    46 Jan  7 20:38 moshme
-rw-r--r-- 1 nonu nonu   655 Jan  7 20:51 .profile
drwx------ 0 nonu nonu   512 Jan 18 15:17 .ssh
-rwxrwxrwx 1 nonu nonu    35 Jan 18 20:12 sshb
-rwxrwxrwx 1 nonu nonu    65 Jan 20 19:40 sshcloudcentos7
-rwxrwxrwx 1 nonu nonu    31 Jan  7 20:38 sshlocalCentOS
-rwxrwxrwx 1 nonu nonu    36 Jan  7 20:38 sshme
-rw-r--r-- 1 nonu nonu     0 Jan  7 20:55 .sudo_as_admin_successful
-rw------- 1 nonu nonu  4459 Feb 19 13:16 .viminfo
```



## 栗子6
输出双引号与**单引号**

```shell
$ echo | awk '{print "\""}'     # 输出双引号
"
$ echo | awk '{print "'\''"}'   # 输出单引号
'
```

