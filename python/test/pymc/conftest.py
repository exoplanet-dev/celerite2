def pytest_configure(config):
    try:
        import pytensor
    except ImportError:
        return

    import platform

    pytensor.config.floatX = "float64"
    pytensor.config.compute_test_value = "raise"
    if platform.system() == "Darwin":
        pytensor.config.gcc.cxxflags = "-Wno-c++11-narrowing"
    config.addinivalue_line("filterwarnings", "ignore")
