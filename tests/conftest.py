"""Shared pytest configuration for the training_hub test suite."""


def pytest_configure(config):
    config.addinivalue_line("markers", "xpu: requires a real Intel XPU (select with -m xpu)")
