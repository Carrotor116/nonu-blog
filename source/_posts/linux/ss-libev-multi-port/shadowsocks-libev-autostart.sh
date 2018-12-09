#!/bin/bash

proc=/usr/local/bin/ss-server
config_dir=/etc/shadowsocks-libev
log_dir=/var/log/shadowcosks

arg=" --fast-open -u -v "

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

mkdir -p $log_dir
for f in ${config_files[@]}
do 
   nohup $proc -c $config_dir/$f $arg  >> $log_dir/${f%.*}.log 2>&1 &
done
