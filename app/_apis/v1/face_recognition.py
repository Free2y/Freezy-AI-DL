# -*- coding:utf-8 -*-

import time
from flask import jsonify, request
from app.common.redprint import Redprint
from app.services.FaceRecService import LivenessCheck
from app.services.OcrService import ImgOcr
from app.common.log_ import logger, error_logger

api = Redprint("fdrService")  # 使用红图的方式

def get_user_info():
    time_now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    uuid = request.form.get('uuid',"")
    name = request.form.get('name',"")
    phone = request.form.get('phone',"")
    id_card = request.form.get('id_card',"")
    sex = request.form.get('sex',"")
    if uuid == '' and name == '' and phone == '' and id_card == '':
        return ''
    return ('_'.join([uuid, name, phone, id_card, sex]),time_now)

@api.route("/checkLiveness", methods=['POST'])
def checkLiveness():
    start_time = time.time()
    video_uri = request.form.get('video_uri')
    img_files = request.files.getlist('imagefiles')
    time_now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

    if video_uri is not None or len(img_files) > 0:
        try:
            liveness = LivenessCheck(img_files, video_uri, get_user_info())
            #print(liveness)
            log_info = {
                'ip': request.remote_addr,
                'return': liveness,
                'time': time_now
            }
            logger.info(jsonify(log_info))
            if video_uri is None:
                video_uri = ''
            return {'code': 0, 'message': '成功',
                 'result': {'liveness': liveness,'video_uri':video_uri,
                          'speed_time': round(time.time() - start_time, 2)}}
        except Exception as ex:
            error_log = {'code': -1, 'message': '产生了一点错误，请检查日志', 'result': str(ex)}
            logger.error(error_logger(time_now,str(ex)), exc_info=True)
            return {'code': 0, 'message': '成功',
                 'result': {'liveness': 'no','video_uri':video_uri,
                          'speed_time': round(time.time() - start_time, 2)}}
    else:
        return {'code': -1, 'message': '请正确传入参数'}
