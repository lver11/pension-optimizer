# conftest.py — pytest configuration for tests/sustainable/
#
# Note: tests/sustainable/__init__.py is intentionally omitted.
# Adding it would turn this directory into a Python package, which causes a
# namespace-shadowing conflict: pytest can no longer find the top-level
# `sustainable` package because the test package intercepts the import.
# The sys.path.insert in test_config.py handles module resolution instead.
