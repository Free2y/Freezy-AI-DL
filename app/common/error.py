# -*- coding:utf-8 -*-

from flask import request, json
from werkzeug.exceptions import HTTPException

class APIException(HTTPException):
    code = 500  # 默认为 500
    msg = "sorry, we make a mistake error"
    error_code = 999  # 表示未知错误

    # 可通过构造函数来改变默认值
    def __init__(self, msg=None, code=None, error_code=None, header=None):
        if code:
            self.code = code
        if error_code:
            self.error_code = error_code
        if msg:
            self.msg = msg
        super(APIException, self).__init__(msg, None)

    def get_body(self, environ=None):
        body = dict(
            msg=self.msg,
            error_code=self.error_code,
            request=request.method + " " + self.get_url_no_param()
        )

        text = json.dumps(body)
        return text

    @staticmethod
    def get_url_no_param():
        full_path = str(request.full_path)
        main_path = full_path.split("?")
        return main_path[0]

    def get_headers(self, environ=None):
        return [("Content-Type","application/json")]
