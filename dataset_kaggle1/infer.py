"""Generate model"""

# Add parent directory to PATH
# https://stackoverflow.com/a/11158224
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from helpers import predict

# File does not exist
predict('model', 'test.csv', 'output.csv', 'output')
