import experiments.launch_docstring
import experiments.launch_induction

# moved the subnetwork probing tests to tests/subnetwork_probing/test_sp_launch.py

def test_acdc_docstring():
    experiments.launch_docstring.main(testing=True)

def test_acdc_induction():
    experiments.launch_induction.main(testing=True)

