import subnetwork_probing.launch_grid_fill
import pytest

@pytest.mark.slow
@pytest.mark.parametrize("reset_networks", [True, False])
def test_sp_grid(reset_networks):
    tasks = ["tracr-reverse", "tracr-proportion", "docstring", "induction", "greaterthan", "ioi"]
    subnetwork_probing.launch_grid_fill.main(TASKS=tasks, job=None, name="sp-test", testing=True, reset_networks=reset_networks)
