import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.run import train_main  # noqa: E402

if __name__ == "__main__":
    train_main("text")
