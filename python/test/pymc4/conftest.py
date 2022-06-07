def pytest_configure(config):
    try:
        import aesara
    except ImportError:
        return

    import platform

    aesara.config.floatX = "float64"
    aesara.config.compute_test_value = "raise"
    if platform.system() == "Darwin":
        aesara.config.gcc.cxxflags = "-Wno-c++11-narrowing"
    config.addinivalue_line("filterwarnings", "ignore")
