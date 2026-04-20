from src.features.extract_features import extract_rule_signals


def test_rule_signal_basics() -> None:
    signals = extract_rule_signals(
        job_description="Data analyst role with immediate joining.",
        company_profile="",
        salary_range="$120000 - $140000",
        contact_info="contact@company.com",
    )

    assert signals.urgent_language is True
    assert signals.salary_missing is False
