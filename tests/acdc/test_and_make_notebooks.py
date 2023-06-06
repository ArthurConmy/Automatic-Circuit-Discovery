"""
This test automatically converts .py notebooks with #%% into .ipynbs and runs them, so that the cell outputs look cool in colab.
"""

import pytest
import subprocess

in_middle_and_out_paths = [
    (
        "notebooks/editing_edges.py",
        "notebooks/_converted/editing_edges.ipynb",
        "notebooks/colabs/ACDC_Editing_Edges_Demo.ipynb",
    ),
    (
        "acdc/main.py",
        "notebooks/_converted/main_demo.ipynb",
        "notebooks/colabs/ACDC_Main_Demo.ipynb",
    ),
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "make_notebooks: mark test to run only when --make-notebooks is given",
    )


@pytest.mark.make_notebook
@pytest.mark.parametrize("in_middle_and_out_paths", in_middle_and_out_paths)
def test_and_make_notebook(in_middle_and_out_paths):
    in_path, middle_path, out_path = in_middle_and_out_paths

    subprocess.run(
        f"""jupytext --to notebook {in_path} -o {middle_path}""".split(), check=True
    )
    subprocess.run(
        f"""papermill {middle_path} {out_path} --kernel=python""".split(), check=True
    )
