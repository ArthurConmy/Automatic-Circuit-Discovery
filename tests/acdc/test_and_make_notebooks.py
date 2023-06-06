"""
This test 
"""

import pytest
import subprocess

in_and_out_paths = [("notebooks/editing_edges.py", "notebooks/colabs/ACDC_Editing_Edges_Demo.ipynb"), ("acdc/main.py", "notebooks/colabs/ACDC_Main_Demo.ipynb")]

@pytest.mark.parametrize("in_and_out_paths", in_and_out_paths)
def test_and_make_notebook(in_and_out_paths):

    in_path, out_path = in_and_out_paths

    subprocess.run(
        f"""jupytext --to notebook {in_path} -o {out_path}""".split(), check=True
    )
    subprocess.run(
        f"""papermill {out_path} {out_path} --kernel python""".split(), check=True
    )