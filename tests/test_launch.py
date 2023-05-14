import subnetwork_probing.launch_train_induction
import subnetwork_probing.launch_sp_docstring
import experiments.launch_docstring
import experiments.launch_induction
import subnetwork_probing.launch_grid_fill

def test_sp_induction():
    subnetwork_probing.launch_train_induction.main(testing=True)


def test_sp_docstring():
    subnetwork_probing.launch_sp_docstring.main(testing=True)

def test_acdc_docstring():
    experiments.launch_docstring.main(testing=True)

def test_acdc_induction():
    experiments.launch_induction.main(testing=True)

def test_sp_grid():
    subnetwork_probing.launch_grid_fill.main(testing=True, use_kubernetes=True)
