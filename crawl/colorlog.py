from logging import handlers
from colorama import Fore, init,Style,Back

class ColorLog:
    init()
    
    pathtologfile = os.path.join('.','logs', 'crawl.txt')

    __logToFile   = logging.getLogger(pathtologfile)
    __logToStdOut = logging.getLogger('.')

    # 设置日志级别
    __logToFile.setLevel(level=logging.INFO)
    __logToStdOut.setLevel(level=logging.INFO)

    # 日志输出格式
    # fmt = logging.Formatter('%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    #logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.ERROR)


    # 输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    # 输出到文件
    # 日志文件按天进行保存，每天一个日志文件
    #file_handler = handlers.TimedRotatingFileHandler(filename=filename, when='D', backupCount=1, encoding='utf-8')
    # 按照大小自动分割日志文件，一旦达到指定的大小重新生成文件
    file_handler = handlers.RotatingFileHandler(filename=pathtologfile, maxBytes=1*1024*1024, backupCount=10, encoding='utf-8')
    file_handler.setFormatter(fmt)

    __logToStdOut.addHandler(console_handler)
    __logToFile.addHandler(file_handler)

    @classmethod
    def debug(cls,msg):
        cls.__logToFile.debug(msg)
        return cls.__logToStdOut.debug(msg)

    @classmethod
    def info(cls, msg):
        cls.__logToFile.info(msg)
        return cls.__logToStdOut.info("{}{}{}".format(Fore.GREEN, msg, Fore.RESET))

    @classmethod
    def warning(cls, msg):
        cls.__logToFile.warning(msg)
        return cls.__logToStdOut.warning("{}{}{}".format(Style.BRIGHT, msg, Style.RESET_ALL))

    @classmethod
    def error(cls, msg):
        cls.__logToFile.error(msg)
        return cls.__logToStdOut.error("{}{}{}{}{}".format(Fore.RED, Style.BRIGHT,  msg, Fore.RESET, Style.RESET_ALL))
    
    @classmethod
    def notice(cls, msg):
        cls.__logToFile.info(msg)
        return cls.__logToStdOut.info("{}{}{}{}{}".format(Fore.RED, Style.BRIGHT,  msg, Fore.RESET, Style.RESET_ALL))

    @classmethod
    def critical(cls, msg):
        cls.__logToFile.critical(msg)
        return cls.__logToStdOut.critical("{}{}{}{}{}".format(Back.RED, Style.BRIGHT,  msg, Back.RESET, Style.RESET_ALL))




if __name__ == '__main__':
    ColorLog.debug("test case begin to run ...")
    ColorLog.info("test case begin to run ...")
    ColorLog.notice("test case begin to run ...")
    ColorLog.warning("test case begin to run ...")
    ColorLog.error("test case begin to run ...")
    ColorLog.critical("test case begin to run ...")
    
    #logging.error("this is common print ...")


    ColorLog.debug("test case begin to run again ...")
    ColorLog.info("test case begin to run again ...")
    ColorLog.notice("test case begin to run again ...")
    ColorLog.warning("test case begin to run again ...")
    ColorLog.error("test case begin to run again ...")
    ColorLog.critical("test case begin to run again ...")
    #logging.error("this is common print agian ...")

    ColorLog.notice("enter any key to exit ...")
