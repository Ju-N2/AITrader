from aitrader.config.settings import Settings


def test_default_settings():
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.upbit_access_key == ""
    assert settings.binance_api_key == ""
