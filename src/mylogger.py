from importlib import reload
import logging

def get_logger(log_title, log_level):
    """
        Initializes the logger
        Check specified log file exist, if not create the directory and file
        Add both file and console handlers
        Arguments
        log_level: Log level for both file and console
    """

    # Logging
    reload(logging)
    # from https://docs.python.org/2/howto/logging-cookbook.html
    global logger

    # Verbose level logger
    logging.VERBOSE = 5
    logging.addLevelName(logging.VERBOSE, "VERBOSE")
    logging.Logger.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)

    logger = logging.getLogger(log_title)
    logger.setLevel(logging.VERBOSE)
    # Log formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    
    # Logging to console
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)
    return logger

def init_logger(log_title='APP', log_level='INFO'):
    # Read Log level
    # Declare predefined set of log levels
    # Check user provided the optional log level parameter, 
    # If so check if its part of predefined list, else fall back to the 
    # default of INFO
    log_levels = {'CRITICAL':50, 'ERROR':40, 'WARNING':30, 'INFO':20, 'DEBUG':10, 'VERBOSE':5, 'NOTSET':0}
    # If environment variable is not set, log at debug level, if incorrect value is set
    # log at info level
    if log_level not in log_levels:
        log_level = 'INFO'
    logLevelNumeric = log_levels[log_level]
    logger = get_logger(log_title, logLevelNumeric)
    logger.critical("Using log level {0}. [This message is raised at CRITICAL level so that this is visible at all log levels]".format(log_level))
    