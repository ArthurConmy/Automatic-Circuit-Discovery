import subnetwork_probing.launch_train_induction
import subnetwork_probing.launch_sp_docstring
import subnetwork_probing.launch_grid_fill
import pytest

def test_sp_induction():
    subnetwork_probing.launch_train_induction.main(testing=True)

def test_sp_docstring():
    subnetwork_probing.launch_sp_docstring.main(testing=True)

# @pytest.mark.skip(reason="TODO get Adria to work on this ")
def test_sp_grid():
    print("WARNING: edited from use_kubernetes=True...")
    subnetwork_probing.launch_grid_fill.main(testing=True)
