from pathlib import Path

from src.pytest_conftest import *


test_root = Path(__file__).parent.resolve()
profile_path = test_root / "profiles"
repo_root = Path(__file__).parent.parent.resolve()
repo_envs = {"REPO_ROOT": str(repo_root)}
register_tests(test_root, profile_path, repo_root, repo_envs)
