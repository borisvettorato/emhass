"""Compatibility shim.

Tests were split into:
- test_pinn_forecast_unit.py
- test_pinn_forecast_integration.py
"""

if __name__ == "__main__":
    import pytest

    pytest.main(["tests_thermal/test_pinn_forecast_unit.py", "tests_thermal/test_pinn_forecast_integration.py", "-v", "-s"])