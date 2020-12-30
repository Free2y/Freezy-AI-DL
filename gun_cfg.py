import multiprocessing

debug = True
loglevel = 'debug'
bind = "0.0.0.0:9876"
pidfile = "logs/gunicorn.pid"
accesslog = "logs/access.log"
errorlog = "logs/debug.log"
# daemon = True
timeout = 60*60*10

# 启动的进程数
worker_class = 'gthread'
workers = multiprocessing.cpu_count()
threads = 300

x_forwarded_for_header = 'X-FORWARDED-FOR'
