from src.features.extract_features import extract_rule_signals, signals_to_list


def test_extract_rule_signals_detects_suspicious_domain_and_missing_fields() -> None:
    signals = extract_rule_signals(
        job_description="Urgent hiring now. No experience needed.",
        company_profile="",
        salary_range="",
        contact_info="Email us at hiring.team@gmail.com",
    )
    findings = signals_to_list(signals)

    assert signals.suspicious_email_domain is True
    assert signals.missing_company_profile is True
    assert signals.salary_missing is True
    assert signals.urgent_language is True
    assert len(findings) >= 3
