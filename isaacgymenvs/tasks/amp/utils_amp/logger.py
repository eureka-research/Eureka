# -----------------------------------------------------------------------------
#   @brief:
#       The logger here will be called all across the project. It is inspired
#   by Yuxin Wu (ppwwyyxx@gmail.com)
#
#   @author:
#       Tingwu Wang, 2017, Feb, 20th
# -----------------------------------------------------------------------------

import logging
import sys
import os
import datetime

__all__ = ['set_file_handler']  # the actual worker is the '_logger'

color2id = {"grey": 30, "red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36, "white": 37}

def colored(text, color):
    return f"\033[{color2id[color]}m{text}\033[0m"



class _MyFormatter(logging.Formatter):
    '''
        @brief:
            a class to make sure the format could be used
    '''

    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'

        if record.levelno == logging.WARNING:
            fmt = date + ' ' + \
                colored('WRN', 'red', attrs=[]) + ' ' + msg
        elif record.levelno == logging.ERROR or \
                record.levelno == logging.CRITICAL:
            fmt = date + ' ' + \
                colored('ERR', 'red', attrs=['underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg

        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt

        return super(self.__class__, self).format(record)


_logger = logging.getLogger('joint_embedding')
_logger.propagate = False
_logger.setLevel(logging.INFO)

# set the console output handler
con_handler = logging.StreamHandler(sys.stdout)
con_handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
_logger.addHandler(con_handler)


class GLOBAL_PATH(object):

    def __init__(self, path=None):
        if path is None:
            path = os.getcwd()
        self.path = path

    def _set_path(self, path):
        self.path = path

    def _get_path(self):
        return self.path


PATH = GLOBAL_PATH()


def set_file_handler(path=None, prefix='', time_str=''):
    # set the file output handler
    if time_str == '':
        file_name = prefix + \
            datetime.datetime.now().strftime("%A_%d_%B_%Y_%I:%M%p") + '.log'
    else:
        file_name = prefix + time_str + '.log'

    if path is None:
        mod = sys.modules['__main__']
        path = os.path.join(os.path.abspath(mod.__file__), '..', '..', 'log')
    else:
        path = os.path.join(path, 'log')
    path = os.path.abspath(path)

    path = os.path.join(path, file_name)
    if not os.path.exists(path):
        os.makedirs(path)

    PATH._set_path(path)
    path = os.path.join(path, file_name)
    from tensorboard_logger import configure
    configure(path)

    file_handler = logging.FileHandler(
        filename=os.path.join(path, 'logger'), encoding='utf-8', mode='w')
    file_handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    _logger.addHandler(file_handler)

    _logger.info('Log file set to {}'.format(path))
    return path


def _get_path():
    return PATH._get_path()


_LOGGING_METHOD = ['info', 'warning', 'error', 'critical',
                   'warn', 'exception', 'debug']

# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
