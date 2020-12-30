# -*- coding:utf-8 -*-

import time
from flask import jsonify, request
from app.common.redprint import Redprint
from app.services.OcrService import ImgOcr
from app.common.log_ import logger, error_logger

api = Redprint("ocrService")  # 使用红图的方式

@api.route("/ocrRecognition", methods=['POST'])
def ocrRecognition():
    start_time = time.time()
    img_files = request.files.getlist('imagefiles')
    img_uri = request.form.get('img_uri')
    time_now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

    if img_uri is not None or len(img_files) > 0:
        try:
            results = ImgOcr(img_files, img_uri)
            log_info = {
                'ip': request.remote_addr,
                'return': results,
                'time': time_now
            }
            logger.info(jsonify(log_info))
            raw = []
            for id,data in results:
                raw.append({'id': id, 'data': data})
            return {'code': 0, 'message': '成功',
                 'result': {'raw': raw,
                          'speed_time': round(time.time() - start_time, 2)}}
        except Exception as ex:
            error_log = {'code': -1, 'message': '产生了一点错误，请检查日志', 'result': str(ex)}
            logger.error(error_logger(time_now,str(ex)), exc_info=True)
            return error_log
    else:
        return {'code': -1, 'message': '没有传入参数'}
