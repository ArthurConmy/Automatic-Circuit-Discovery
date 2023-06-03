import subnetwork_probing.launch_train_induction
import subnetwork_probing.launch_sp_docstring
import subnetwork_probing.launch_grid_fill
import pytest

@pytest.mark.skip(reason="Extremely slow - TODO ask Adria if this is supposed to a) be on CPU and b) run 10 runs (I think)")
def test_sp_induction():
    subnetwork_probing.launch_train_induction.main(testing=True)

@pytest.mark.skip(reason="Extremely slow - TODO ask Adria if this is supposed to a) be on CPU and b) run 10 runs (I think)")
def test_sp_docstring():
    subnetwork_probing.launch_sp_docstring.main(testing=True)

@pytest.mark.skip(reason="TODO ask Adria why this suddenly needs 4 arguments...?")
def test_sp_grid():
    subnetwork_probing.launch_grid_fill.main(testing=True, use_kubernetes=True)