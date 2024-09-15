import pytest

def pytest_addoption(parser):
    # Add custom command-line options
    parser.addoption(
        "--encoder-name",
        action="store",
        default=None,
        help="Specify the encoder name to test"
    )
    
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Specify the device to use (default: cpu)"
    )

@pytest.fixture
def encoder_name(request):
    # Access the value of the custom option for encoder name
    return request.config.getoption("--encoder-name")

@pytest.fixture
def device(request):
    # Access the value of the custom option for device
    return request.config.getoption("--device")
