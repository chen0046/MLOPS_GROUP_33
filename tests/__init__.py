import os
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

print(f"_TEST_ROOT: {_TEST_ROOT}")
print(f"_PROJECT_ROOT: {_PROJECT_ROOT}")
print(f"_PATH_DATA: {_PATH_DATA}")