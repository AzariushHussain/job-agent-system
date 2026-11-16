import re
import smtplib
import uuid
from typing import List, Optional, Callable, Literal
from urllib.parse import urlparse

import dns.resolver
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# ATS platforms, job boards, and non-company domains that must never be used for email inference
ATS_AND_IGNORED_DOMAINS = {
    "example.com",
    "sentry.io",
    "w3.org",
    "schema.org",
    "linkedin.com",
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "workday.com",
    "myworkdayjobs.com",
    "workable.com",
    "breezy.hr",
    "smartrecruiters.com",
    "bamboohr.com",
    "jobvite.com",
    "icims.com",
    "wellfound.com",
    "angel.co",
    "naukri.com",
    "instahyre.com",
    "indeed.com",
    "glassdoor.com",
    "ycombinator.com",
}


class EmailVerificationResult(BaseModel):
    email: str
    source: Literal["jd_explicit", "domain_inferred"]
    is_valid: bool
    status: Literal[
        "smtp_verified",
        "mx_verified",
        "catch_all",
        "invalid_syntax_or_ats",
        "invalid_no_mx",
        "smtp_rejected",
        "unverified_inferred_dropped",
    ]
    mx_host: Optional[str] = None


class ScrapeResult(BaseModel):
    success: bool
    text: str = ""
    discovered_emails: List[str] = Field(default_factory=list)
    verification_report: List[EmailVerificationResult] = Field(default_factory=list)
    error_message: Optional[str] = None
    source_tier: str = "raw_input"


def is_blocked_or_ats_domain(domain: str) -> bool:
    """Returns True if the domain or its parent domain is a known ATS, job board, or placeholder."""
    clean = domain.lower().strip(".")
    if not clean or "." not in clean:
        return True
    for blocked in ATS_AND_IGNORED_DOMAINS:
        if clean == blocked or clean.endswith("." + blocked):
            return True
    return False


def resolve_mx_hosts(domain: str, timeout: float = 3.0) -> List[str]:
    """
    Queries DNS MX records for `domain` and returns mail exchangers sorted by priority.
    Returns an empty list if the domain has no MX records, Null MX (RFC 7505), or NXDOMAIN.
    """
    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = timeout
        answers = resolver.resolve(domain, "MX")
        records = []
        for rdata in answers:
            exchange = str(rdata.exchange).rstrip(".")
            # Ignore RFC 7505 Null MX ('.' or empty)
            if exchange:
                records.append((rdata.preference, exchange))
        records.sort(key=lambda x: x[0])
        return [host for _, host in records]
    except Exception:
        return []


def probe_smtp_mailbox(
    email: str, mx_host: str, timeout: float = 3.5
) -> Literal["smtp_verified", "catch_all", "smtp_rejected", "smtp_unreachable"]:
    """
    Performs a non-sending SMTP RCPT TO handshake against `mx_host` and checks
    whether the server is a catch-all. Never sends a DATA payload.
    """
    domain = email.split("@")[-1]
    random_probe = f"_verify_probe_{uuid.uuid4().hex[:8]}@{domain}"

    try:
        with smtplib.SMTP(mx_host, 25, timeout=timeout) as smtp:
            smtp.ehlo_or_helo_if_needed()
            smtp.mail("verify@job-agent.local")
            code, _ = smtp.rcpt(email)

            if code in (250, 251):
                # Probe a random nonexistent address to detect catch-all servers
                probe_code, _ = smtp.rcpt(random_probe)
                if probe_code in (250, 251):
                    return "catch_all"
                return "smtp_verified"

            if 500 <= code < 600:
                return "smtp_rejected"

            return "smtp_unreachable"
    except Exception:
        # Port 25 is frequently blocked on residential ISPs / cloud VPCs
        return "smtp_unreachable"


def verify_email_address(
    email: str,
    source: Literal["jd_explicit", "domain_inferred"] = "jd_explicit",
    check_smtp: bool = True,
    mx_resolver: Optional[Callable[[str], List[str]]] = None,
    smtp_prober: Optional[Callable[[str, str], str]] = None,
) -> EmailVerificationResult:
    """
    Verifies an email address using Syntax + ATS filtering, DNS MX lookup, and optional SMTP probing.
    - `jd_explicit` emails pass if they have valid MX records and are not SMTP-rejected.
    - `domain_inferred` emails ONLY pass if `smtp_verified` (never on catch-all or MX-only).
    """
    clean_email = email.lower().strip(".,;: ")
    if (
        not EMAIL_REGEX.fullmatch(clean_email)
        or clean_email.endswith((".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"))
    ):
        return EmailVerificationResult(
            email=clean_email,
            source=source,
            is_valid=False,
            status="invalid_syntax_or_ats",
        )

    domain = clean_email.split("@")[-1]
    if is_blocked_or_ats_domain(domain):
        return EmailVerificationResult(
            email=clean_email,
            source=source,
            is_valid=False,
            status="invalid_syntax_or_ats",
        )

    # Stage 2: DNS MX Lookup
    resolve_fn = mx_resolver or resolve_mx_hosts
    mx_hosts = resolve_fn(domain)
    if not mx_hosts:
        return EmailVerificationResult(
            email=clean_email,
            source=source,
            is_valid=False,
            status="invalid_no_mx",
        )

    primary_mx = mx_hosts[0]

    # Stage 3: SMTP Handshake Probe
    if check_smtp:
        probe_fn = smtp_prober or probe_smtp_mailbox
        smtp_status = probe_fn(clean_email, primary_mx)

        if smtp_status == "smtp_rejected":
            return EmailVerificationResult(
                email=clean_email,
                source=source,
                is_valid=False,
                status="smtp_rejected",
                mx_host=primary_mx,
            )

        if smtp_status == "smtp_verified":
            return EmailVerificationResult(
                email=clean_email,
                source=source,
                is_valid=True,
                status="smtp_verified",
                mx_host=primary_mx,
            )

        if smtp_status == "catch_all":
            # Keep explicit JD emails on catch-all domains, but drop guessed/inferred ones
            is_ok = source == "jd_explicit"
            return EmailVerificationResult(
                email=clean_email,
                source=source,
                is_valid=is_ok,
                status="catch_all" if is_ok else "unverified_inferred_dropped",
                mx_host=primary_mx,
            )

    # Fallback when SMTP Port 25 is unreachable or disabled:
    # Accept emails explicitly written in the JD (since MX is valid), but drop guessed emails.
    if source == "jd_explicit":
        return EmailVerificationResult(
            email=clean_email,
            source=source,
            is_valid=True,
            status="mx_verified",
            mx_host=primary_mx,
        )

    return EmailVerificationResult(
        email=clean_email,
        source=source,
        is_valid=False,
        status="unverified_inferred_dropped",
        mx_host=primary_mx,
    )


def extract_and_verify_emails(
    text: str,
    job_url: str = "",
    infer_if_missing: bool = True,
    check_smtp: bool = True,
    mx_resolver: Optional[Callable[[str], List[str]]] = None,
    smtp_prober: Optional[Callable[[str, str], str]] = None,
) -> tuple[List[str], List[EmailVerificationResult]]:
    """
    Extracts emails from JD text, optionally infers `careers@` / `founders@` from company domain,
    and verifies every candidate before returning valid addresses.
    """
    candidates: List[tuple[str, Literal["jd_explicit", "domain_inferred"]]] = []
    seen = set()

    # 1. Explicit regex matches from JD text
    for match in EMAIL_REGEX.findall(text):
        clean = match.lower().strip(".,;:")
        if clean not in seen:
            seen.add(clean)
            candidates.append((clean, "jd_explicit"))

    # 2. Only attempt domain inference if no explicit emails were found AND domain is not an ATS
    if not candidates and infer_if_missing and job_url:
        parsed = urlparse(job_url)
        host = parsed.netloc.lower().removeprefix("www.")
        if host and not is_blocked_or_ats_domain(host):
            for prefix in ("careers", "founders"):
                inferred = f"{prefix}@{host}"
                if inferred not in seen:
                    seen.add(inferred)
                    candidates.append((inferred, "domain_inferred"))

    verified_emails: List[str] = []
    reports: List[EmailVerificationResult] = []

    for email_addr, source_type in candidates:
        res = verify_email_address(
            email=email_addr,
            source=source_type,
            check_smtp=check_smtp,
            mx_resolver=mx_resolver,
            smtp_prober=smtp_prober,
        )
        reports.append(res)
        if res.is_valid:
            verified_emails.append(res.email)

    return verified_emails, reports


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
    retry=retry_if_exception_type(httpx.HTTPError),
    reraise=True,
)
def _fetch_url_with_retry(url: str, timeout: float = 10.0) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) JobAgentScout/1.0"}
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def scrape_job_description(
    job_url: str,
    existing_raw_text: str = "",
    http_fetcher: Optional[Callable[[str], str]] = None,
    mx_resolver: Optional[Callable[[str], List[str]]] = None,
    smtp_prober: Optional[Callable[[str, str], str]] = None,
    check_smtp: bool = True,
) -> ScrapeResult:
    """
    Resilient 3-tier JD scraper with DNS MX + SMTP email verification.
    """
    fetcher = http_fetcher or _fetch_url_with_retry

    # Tier 1: Raw text already supplied
    if existing_raw_text and len(existing_raw_text.strip()) >= 40:
        clean_text = existing_raw_text.strip()
        emails, reports = extract_and_verify_emails(
            clean_text,
            job_url=job_url,
            check_smtp=check_smtp,
            mx_resolver=mx_resolver,
            smtp_prober=smtp_prober,
        )
        return ScrapeResult(
            success=True,
            text=clean_text,
            discovered_emails=emails,
            verification_report=reports,
            source_tier="raw_input",
        )

    if not job_url or not job_url.startswith(("http://", "https://")):
        return ScrapeResult(
            success=False,
            text=existing_raw_text or "",
            error_message="Missing valid job_url and raw_jd_text is empty.",
            source_tier="failed",
        )

    # Tier 2: Jina Reader API for clean Markdown
    try:
        jina_url = f"https://r.jina.ai/{job_url}"
        md_text = fetcher(jina_url)
        if md_text and len(md_text.strip()) >= 80:
            clean_md = md_text.strip()
            emails, reports = extract_and_verify_emails(
                clean_md,
                job_url=job_url,
                check_smtp=check_smtp,
                mx_resolver=mx_resolver,
                smtp_prober=smtp_prober,
            )
            return ScrapeResult(
                success=True,
                text=clean_md,
                discovered_emails=emails,
                verification_report=reports,
                source_tier="jina_reader",
            )
    except Exception:
        pass

    # Tier 3: Direct HTML Fetch + BeautifulSoup
    try:
        raw_html = fetcher(job_url)
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        extracted_text = soup.get_text(separator="\n", strip=True)
        if len(extracted_text) >= 40:
            emails, reports = extract_and_verify_emails(
                extracted_text,
                job_url=job_url,
                check_smtp=check_smtp,
                mx_resolver=mx_resolver,
                smtp_prober=smtp_prober,
            )
            return ScrapeResult(
                success=True,
                text=extracted_text,
                discovered_emails=emails,
                verification_report=reports,
                source_tier="direct_html",
            )
        return ScrapeResult(
            success=False,
            text=extracted_text,
            error_message="Scraped page contained insufficient text (< 40 chars).",
            source_tier="failed",
        )
    except Exception as exc:
        return ScrapeResult(
            success=False,
            text="",
            error_message=f"All scraper tiers failed for {job_url}: {exc}",
            source_tier="failed",
        )
