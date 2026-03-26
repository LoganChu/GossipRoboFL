"""
conftest.py — pytest configuration for GossipRoboFL.

Run tests with ml_env:
    C:/Users/s-lch/anaconda3/envs/ml_env/python.exe -m pytest tests/ -v -m "not slow"

Or set up a .pytest-env / tox.ini to point at the right interpreter.
"""
import sys
import os

# Ensure the project root is on sys.path so `from src.xxx import ...` works
# regardless of where pytest is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
