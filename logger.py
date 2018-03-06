import logging

# logging.basicConfig(filename="cipher.log", level=logging.INFO)
# logger = logging.getLogger(__name__)

logger = logging.getLogger('cipherbreak')
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

# create the logging file handler
fh = logging.FileHandler("cipher.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add handler to logger object
logger.addHandler(fh)
