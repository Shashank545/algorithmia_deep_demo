from . import mnist_model_build


def test_deep_sense_deploy_demo():
    assert mnist_model_build.mnist_model_inference("mnist_test.PNG") == 6