# -*- coding:utf-8 -*-

from datetime import date
import numpy as np
from flask import Flask as _Flask
from flask.json import JSONEncoder as _JSONEncoder

from app.common.error_code import ServerError


class JSONEncode(_JSONEncoder):

    def default(self, o):
        # 判断是否存在keys 和 __getitem__
        if hasattr(o, "keys") and hasattr(o, "__getitem__"):
            return dict(o)
        # 处理不能序列化的数据类型
        if isinstance(o, date):
            return o.strftime("%Y-%m-%d")
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise ServerError()


class Flask(_Flask):
    # 使用我们自己定义的JSONEncode
    json_encoder = JSONEncode

def register_bluprints(app):
    """注册蓝图
    :param app: flask实例
    :return:
    """

    # 使用公用的的蓝图
    from app._apis.v1.register_service import create_blueprint
    # 添加前缀
    app.register_blueprint(create_blueprint(), url_prefix="/v1")



def create_app():
    # 创建app
    app = Flask(__name__)
    # 加载配置文件
    # app.config.from_object("app.config.setting")
    # app.config.from_object("app.config.secure")

    # 注册
    register_bluprints(app)

    return app
