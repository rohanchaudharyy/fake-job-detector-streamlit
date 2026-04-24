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


def _contains_salary_info(text: str) -> bool:
    salary_patterns = [
        r"\$\s?\d[\d,]*(\.\d+)?(\s?-\s?\$\s?\d[\d,]*(\.\d+)?)?",
        r"\b\d{2,3}k\b",
        r"\bsalary\b",
        r"\bper hour\b",
        r"\bper annum\b",
        r"\bper year\b",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in salary_patterns)


def _contains_company_context(text: str) -> bool:
    hints = [
        "about us",
        "about the company",
        "company overview",
        "we are",
        "our company",
        "join our team",
    ]
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


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
    has_company_info = len(company.strip()) > 0 or _contains_company_context(description)
    has_salary_info = len(salary.strip()) > 0 or _contains_salary_info(description)

    return RuleSignals(
        suspicious_email_domain=email_domain in SUSPICIOUS_EMAIL_DOMAINS,
        missing_company_profile=not has_company_info,
        salary_missing=not has_salary_info,
        urgent_language=_contains_urgent_language(description),
    )


def signals_to_list(signals: RuleSignals) -> list[str]:
    output = []
    if signals.suspicious_email_domain:
        output.append("Contact email uses a common free email domain.")
    if signals.missing_company_profile:
        output.append("Company profile is not clearly provided.")
    if signals.salary_missing:
        output.append("Salary details are not clearly provided.")
    if signals.urgent_language:
        output.append("Description contains urgency or low-barrier hiring language.")
    return output
