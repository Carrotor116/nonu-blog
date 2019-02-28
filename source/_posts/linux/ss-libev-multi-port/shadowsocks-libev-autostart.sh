#!/bin/bash

proc=/usr/local/bin/ss-server
config_dir=/etc/shadowsocks-libev
log_dir=/etc/shadowsocks-libev

arg=" -v "

config_files=()
files=$(ls ${config_dir}/config_*.json)

for f in ${files[@]}
do 
   fn=${f##*/}
   nohup $proc -c $f $arg  >> ${log_dir}/${fn%.*}.log 2>&1 &
done
