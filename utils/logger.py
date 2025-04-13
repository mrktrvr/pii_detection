'''
logger.py

logging module

# Usage
## import
CDIR = os.path.abspath(os.path.dirname(__file__))
UTIL_DIR = os.path.join(CDIR, RELATIVE_PATH_TO_UTILS_DIR)
sys.path.append(UTIL_DIR)
from logger import logger

## logger
logger.debug(str)
logger.info(str)
logger.warning(str)
logger.error(str)
logger.critial(str)

## set log level (info)
log_level = 'INFO'
logger.setLevel(log_level)
logger.debug('This won't be seen.')
logger.info('This will be seen.')
logger.warning('This will be seen.')
logger.error('This will be seen.')

## set log level (warning)
log_level = 'WARNING'
logger.setLevel(log_level)
logger.debug('This will NOT be seen.')
logger.info('This will NOT be seen.')
logger.warning('This will be seen.')
logger.error('This will be seen.')

### log levels

### log level list
- 'CRITICAL'
- 'ERROR'
- 'WARNING'
- 'INFO'
- 'DEBUG'

### default log level
- INFO
'''

import os
import sys
from logging import Filter
from logging import getLogger
from logging import StreamHandler
from logging import Formatter
from logging import FileHandler
from logging import getLoggerClass
from collections import namedtuple

CDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CDIR)

config_dat = {
    'level': 'INFO',
    'datefmt': '%Y-%m-%d %H:%M:%S',
    'fmt_separator': '|',
    'additional_attrs': {},
    'module_name': os.getenv('WEBSITE_SITE_NAME', ''),
    'log_table_name': 't_error_log',
    'module_no_clm': 'module_no',
    'source_column_name': 'module_name',
}
config_dat['format'] = [
    '%(asctime)s',
    '%(levelname)s',
    '%(filename)s(%(lineno)d)',
    '%(funcName)s',
    '%(message)s',
]

Config = namedtuple('config', config_dat)
CONFIG = Config(**config_dat)

loggers = {}


class Logger(getLoggerClass()):
    '''
    Logger class

    logger_cls = Logger()
    '''

    def __init__(self, **args):
        '''
        @Args - All optional
        name: logger name, str, default __name__
        level: log level, str, default level
                  'debug', 'info', 'warn', 'error' or 'clitical'
        datefmt: date time format, str, default, '%Y-%m-%d %H:%M:%S'
        additional_attrs: additional attributes, dict,
                          {name: value, ...}
        '''
        self._name = args.get('name', __name__)
        self._level = args.get('level', CONFIG.level)
        self._datefmt = args.get('datefmt', CONFIG.datefmt)
        self._fmt_separator = args.get('fmt_separator', CONFIG.fmt_separator)
        self._additional_attrs = args.get('additional_attrs',
                                          CONFIG.additional_attrs)

        self._format_list = CONFIG.format

        global loggers
        self._loggers = loggers

        self._formatter = None
        self._logger = None

        self._init_logger()
        self._init_stream_handler()

    def __del__(self):
        pass

    def logger(self):
        return self._logger

    def _init_logger(self):
        # --- global logger container
        self._logger = loggers.get(self._name, None)
        if self._logger is not None:
            map(self._logger.removeHandler, self._logger.handlers[:])
            map(self._logger.removeFilter, self._logger.filters[:])

        # --- loger object
        self._logger = getLogger(self._name)
        self.setLevel(self._level)

        # --- logger handlear initialisation
        if self._logger.handlers:
            self._logger.handlers = []

        # --- global logger container
        self._loggers.update({self._name: self._logger})

        # --- filter
        self._init_filter()

        # --- formatter
        self._init_formatter()

    def _init_formatter(self):
        # --- formatter
        format_str = self._fmt_separator.join(self._format_list)
        self._formatter = Formatter(format_str, datefmt=self._datefmt)

    def _init_filter(self):
        self.add_filter(self._additional_attrs)

    def _init_stream_handler(self):
        # --- stream handler
        handler = StreamHandler()
        handler.setFormatter(self._formatter)
        self.add_handler(handler)

    def add_filter(self, additional_attrs):
        for attr_name, attr_value in additional_attrs.items():
            filt = CustomAttributeFilter(attr_value, attr_name)
            self._logger.addFilter(filt)
            attr_format = filt.attr_format()
            if attr_format != '':
                self._format_list.insert(2, attr_format)

    def add_handler(self, handler):
        self._logger.addHandler(handler)

    def init_file_handler(self, log_filepath):
        # --- file handler
        handler = FileHandler(log_filepath)
        handler.setFormatter(self._formatter)
        self._logger.addHandler(handler)


class CustomAttributeFilter(Filter):
    '''
    '''

    def __init__(self, attr_value: str, attr_name: str, **args: int):
        '''
        attr_value: attribute value
        attr_name: attribute name
        '''
        self._attr_value = attr_value
        self._attr_name = attr_name
        self._attr_format = ''

    def filter(self, record):
        '''
        filter method override
        '''
        if self._attr_value != '':
            record.__dict__.update({self._attr_name: self._attr_value})
        return True

    def attr_format(self):
        '''
        Returns format of this filter for Formatting
        '''
        if self._attr_value != '':
            self._attr_format = ''.join(['%(', self._attr_name, ')s'])
        return self._attr_format


logger_cls = Logger()
logger = logger_cls.logger()


def main():
    e = Exception('test exception')
    logger.error(e)
    log_levels = [
        'CRITICAL',
        'ERROR',
        'WARNING',
        # 'INFO',
        # 'DEBUG',
    ]
    loggers = [
        # logger.debug,
        # logger.info,
        logger.warning,
        logger.error,
        # logger.critical,
        # logger.exception,
    ]

    for log_lvl in log_levels:
        logger.setLevel(log_lvl)
        for module in loggers:
            module(', '.join([
                '--- test log ---',
                'Log level: %s' % log_lvl,
                'Method: %s' % module.__name__,
            ]))


if __name__ == '__main__':
    main()
