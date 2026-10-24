import importlib.metadata

import name_that_llm


def test_package_imports_and_reports_its_version():
    # The version comes from installed metadata, so it can never drift from pyproject.toml.
    assert name_that_llm.__version__ == importlib.metadata.version("name-that-llm")
