import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from common.run import test_main  # noqa: E402

if __name__ == "__main__":
    test_main("text")
