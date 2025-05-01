import logging
import os
import sys
from datetime import datetime

# ==============================BASIC SETUP ===========================================================

# Setup logging
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
	level=logging.INFO,
	filename=os.path.join(LOG_DIR, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"),
	format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)


# Custom exception class
class CustomException(Exception):
	def __init__(self, msg, details: sys):
		_, _, tb = details.exc_info()
		self.msg = f"Error in [{tb.tb_frame.f_code.co_filename}] at line [{tb.tb_lineno}]: {msg}"
	def __str__(self): return self.msg

# ====================================================================================================