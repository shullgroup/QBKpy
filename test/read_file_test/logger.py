    
import logging
import logging.config
# Get the logger specified in the file
logger = logging.getLogger(__name__)

def setup_logging():
    '''Setup logging configuration'''

    try:
        logging.config.dictConfig(logger_config)
    except Exception as e:
        print('Setting logger failed! Use default logging level!')
        logging.basicConfig(level='INFO')
        print(e)





logger_config ={
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(lineno)s - %(message)s',
        },
    },    
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'ERROR',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',
        },    
        'info_file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'filename': 'info.log',
            'maxBytes': 1048576,
            'backupCount': 1,
            'encoding': 'utf8',
        },    
        'error_file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'simple',
            'filename': 'err.log',
            'maxBytes': 1048576,
            'backupCount': 1,
            'encoding': 'utf8',
        }
    },    
    'loggers': {
        'console_logger': {
            # 'level': 'WARNING',
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False,
        }
    },    
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'info_file_handler', 'error_file_handler'],
        # 'handlers': ['console', 'error_file_handler'],
    },
}