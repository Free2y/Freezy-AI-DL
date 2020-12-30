# -*- coding:utf-8 -*-

from flask import Blueprint
from app._apis.v1 import face_recognition,ocr

def create_blueprint():
    bp_v1 = Blueprint("v1", __name__)

    # 红图向蓝图的注册
    face_recognition.api.register(bp_v1)
    ocr.api.register(bp_v1)

    return bp_v1
