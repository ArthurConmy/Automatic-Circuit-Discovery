import torch
from acdc.greaterthan.utils import greaterthan_metric_reference, greaterthan_metric, get_year_data
from acdc.ioi.utils import get_gpt2_small


def test_greaterthan_metric():
    model = get_gpt2_small(device="cpu")
    data, _ = get_year_data(20, model)
    logits = model(data)

    expected = greaterthan_metric_reference(logits, data)
    actual = greaterthan_metric(logits, data)
    torch.testing.assert_close(actual, torch.as_tensor(expected))
