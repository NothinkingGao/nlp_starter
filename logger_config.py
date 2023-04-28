import sys
import logging
import logging.handlers
import datetime

# 日期格式
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

# 添加日志器的名称标识
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)


all_handler = logging.handlers.TimedRotatingFileHandler(
    filename='all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0)
)
all_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s")
)

# 配置一个流处理器,用来显示在控制台
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s")
)


err_handler = logging.FileHandler('error.log')
err_handler.setLevel(logging.ERROR)

# 格式器
err_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d -%(pathname)s \n%(message)s")
)

# 给logger 添加处理器
logger.addHandler(stream_handler)
logger.addHandler(all_handler)
logger.addHandler(err_handler)

# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warning message')
# logger.error('error message')
# logger.critical('critical message')
# logger.info('日志输出完成')