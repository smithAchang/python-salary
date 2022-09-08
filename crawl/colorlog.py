#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from colorama import Fore, init,Style,Back

class ColorLog:
    init()

    @classmethod
    def debug(cls,msg):
        return logging.debug(msg)

    @classmethod
    def info(cls, msg):
        return logging.info("{}{}{}".format(Fore.GREEN, msg, Fore.RESET))

    @classmethod
    def warning(cls, msg):
        return logging.warning("{}{}{}".format(Style.BRIGHT, msg, Style.RESET_ALL))

    @classmethod
    def error(cls, msg):
        return logging.error("{}{}{}{}{}".format(Fore.RED, Style.BRIGHT,  msg, Fore.RESET, Style.RESET_ALL))

    @classmethod
    def critical(cls, msg):
        return logging.critical("{}{}{}{}{}".format(Back.RED, Style.BRIGHT,  msg, Back.RESET, Style.RESET_ALL))




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
    LOG = ColorLog()
    LOG.debug("test case begin to run ...")
    LOG.info("test case begin to run ...")
    LOG.warning("test case begin to run ...")
    LOG.error("test case begin to run ...")
    LOG.critical("test case begin to run ...")
   
    logging.error("this is common print ...")
    LOG.debug("test case begin to run again ...")
    LOG.info("test case begin to run again ...")
    LOG.warning("test case begin to run again ...")
    LOG.error("test case begin to run again ...")
    LOG.critical("test case begin to run again ...")



    
