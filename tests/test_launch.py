import experiments.launch_docstring
import experiments.launch_induction
import pytest

# moved the subnetwork probing tests to tests/subnetwork_probing/test_sp_launch.py

@pytest.mark.skip(reason="Extremely slow - TODO ask Adria if this is supposed to a) be on CPU and b) run 10 runs (I think)")
def test_acdc_docstring():
    experiments.launch_docstring.main(testing=True)

@pytest.mark.skip(reason="Extremely slow - TODO ask Adria if this is supposed to a) be on CPU and b) run 10 runs (I think)")
def test_acdc_induction():
    experiments.launch_induction.main(testing=True)