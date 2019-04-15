import logging
import os

class Logger:
    def __init__(self):
        """
        Initialize Logger.
        """

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
    
    def initialize_log(self, filename):
        # AP Result Logging
        resulthandler = logging.FileHandler(os.path.join(os.getcwd(), filename))
        resulthandler.setLevel(logging.INFO)
        filefmt = logging.Formatter('%(asctime)s: %(message)s')
        resulthandler.setFormatter(filefmt)
        
        # Debug logging
        debuger = logging.StreamHandler()
        debuger.setLevel(logging.DEBUG)
        filefmt = logging.Formatter('%(filename)s-%(lineno)d: %(message)s')
        debuger.setFormatter(filefmt)
        debugerfilter = logging.Filter()
        debugerfilter.filter = lambda record: record.levelno < logging.WARNING
        debuger.addFilter(debugerfilter)

        # Add Handlers
        self.logger.addHandler(resulthandler)
        self.logger.addHandler(debuger)

log = Logger()
