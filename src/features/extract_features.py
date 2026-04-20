import re
from dataclasses import dataclass


SUSPICIOUS_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}


@dataclass
class RuleSignals:
    suspicious_email_domain: bool
    missing_company_profile: bool
    salary_missing: bool
    urgent_language: bool


def _contains_urgent_language(text: str) -> bool:
    keywords = [
        "urgent hiring",
        "immediate joining",
        "quick money",
        "no experience needed",
        "limited slots",
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)


def _extract_email_domain(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    if not match:
        return ""
    return match.group(1).lower()


def extract_rule_signals(
    job_description: str,
    company_profile: str = "",
    salary_range: str = "",
    contact_info: str = "",
) -> RuleSignals:
    description = job_description or ""
    company = company_profile or ""
    salary = salary_range or ""
    contact = contact_info or ""

    combined_text = " ".join([description, contact])
    email_domain = _extract_email_domain(combined_text)

    return RuleSignals(
        suspicious_email_domain=email_domain in SUSPICIOUS_EMAIL_DOMAINS,
        missing_company_profile=len(company.strip()) == 0,
        salary_missing=len(salary.strip()) == 0,
        urgent_language=_contains_urgent_language(description),
    )


def signals_to_list(signals: RuleSignals) -> list[str]:
    output = []
    if signals.suspicious_email_domain:
        output.append("Contact email uses a common free email domain.")
    if signals.missing_company_profile:
        output.append("Company profile is missing.")
    if signals.salary_missing:
        output.append("Salary range is missing.")
    if signals.urgent_language:
        output.append("Description contains urgency or low-barrier hiring language.")
    return output
