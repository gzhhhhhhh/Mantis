def run_net(*args, **kwargs):
    raise RuntimeError(
        "Pre-training code has been removed from the inference-only release."
    )


def test_net(*args, **kwargs):
    raise RuntimeError(
        "Pre-training utilities are not available in the inference-only release."
    )
