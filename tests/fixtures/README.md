# Test Fixtures

This directory contains sample data and configurations for EXMO gait analysis tests.

## Files

### Configuration Files
- `sample_config_v11.yaml`: v1.1.0 legacy configuration
- `sample_config_v12.yaml`: v1.2.0 calibrated configuration

### Sample Data
Sample CSV data files are dynamically generated in tests using numpy fixtures.
This approach ensures:
- No large binary files in repository
- Reproducible test data (seeded random generation)
- Easy customization for specific test scenarios

## Usage

Fixtures are automatically loaded by pytest through `conftest.py`.

Test code can access fixtures like:

```python
def test_my_feature(sample_trajectory_2d, sample_likelihood_high):
    # Use fixtures in test
    assert len(sample_trajectory_2d) == 100
```

## Adding New Fixtures

Add new fixtures to `/tests/conftest.py` to make them available across all tests.
