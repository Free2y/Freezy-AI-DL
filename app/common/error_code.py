# -*- coding:utf-8 -*-

from .error import APIException

class Success(APIException):
    code = 201
    msg = "ok"
    error_code = 0


class DeleteSuccess(Success):
    code = 202
    error_code = 1


class ServerError(APIException):
    code = 500
    msg = "sorry, we make a mistake error)"
    error_code = 999


class ClientTypeError(APIException):
    """
    常见错误码:
    400 401 403 404
    500
    200 201 204
    301 302
    """
    code = 400  # 请求参数错误
    msg = "client is invalid"
    error_code = 1006


class ParameterException(APIException):
    code = 400
    msg = "invalid parameter"
    error_code = 1000


class NotFound(APIException):
    code = 404
    msg = "the resource are not_found 0_0~~~"
    error_code = 1001


class AuthFailed(APIException):
    code = 401
    error_code = 1005
    msg = "authorization failed"


class Forbidden(APIException):
    code = 403
    error_code = 1004
    msg = "forbidden,not in scope"


class DuplicateGift(APIException):
    code = 400
    error_code = 2001
    msg = 'the current book has already in gift'
