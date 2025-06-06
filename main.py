import os
import sys

from src.api.server import run_server

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8192)
