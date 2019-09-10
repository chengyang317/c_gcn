# coding=utf-8
import logging
import sys

sys_logger = logging.getLogger('pt_pack')
sys_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)
console_handler.setFormatter(formatter)

sys_logger.addHandler(console_handler)