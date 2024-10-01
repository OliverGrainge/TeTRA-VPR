import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NeuroPress.dataloaders.QDistill import QVPRDistill

ob = QVPRDistill()

print(ob)
