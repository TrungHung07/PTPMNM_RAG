# conftest.py — pytest configuration
# Thêm thư mục gốc vào sys.path để import src.* trong test files
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
