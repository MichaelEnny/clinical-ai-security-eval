import sys
from pathlib import Path

# Make the task module importable from anywhere pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent / "task"))
