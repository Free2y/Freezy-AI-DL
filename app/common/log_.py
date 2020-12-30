# -*- coding:utf-8 -*-

import logging
import time
import os
import sys

# 日志管理
from flask import request, jsonify

logger = logging.getLogger('log.' + __name__)
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = './logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_name = log_path + rq + '.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)

def error_logger(time, ex):
    error_logger = {}
    error_logger['ip'] = str(request.remote_addr)
    error_logger['time'] = time
    error_logger['api'] = str(request.path)
    error_logger['method'] = str(request.method)
    error_logger['headers'] = str(request.headers).rstrip()
    error_logger['get_args'] = str(request.args)
    error_logger['post_args'] = str(request.form)
    error_logger['file_args'] = str(request.files)
    error_logger['ex_info'] = ex
    return jsonify(error_logger)
