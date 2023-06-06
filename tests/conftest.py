"""
Added to allow selective running of tests
"""

import pytest

def pytest_addoption(parser):
    parser.addoption("--make-notebook", action="store_true", default=False,
                     help="Run notebook tests")

@pytest.fixture
def make_notebooks(request):
    return request.config.getoption("--make-notebook")

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "make_notebooks: mark test to run only when --make-notebook is given"
    )

def pytest_runtest_setup(item):
    make_notebook_marker = item.get_closest_marker("make_notebook")
    if make_notebook_marker is not None and not item.config.getoption("--make-notebook"):
        pytest.skip("need --make-notebooks option to run")