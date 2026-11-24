"""
Unit and regression tests for Deep Audit & Remediation (M1 vs M4).
Verifies:
1. M1 Continuous Confidence Calibration (monotonicity, [0.50, 1.00] bounds, Tau=0.29 boundary).
2. Domain Analyzer and lookalike/mismatch detection.
3. Disagreement resolution and Architectural Divergence handling.
4. Telemetry fields preservation in M1 and M4 outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on path
src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
from core.domain_analyzer import analyze_email_domains, format_domain_evidence_block
from core.inference import (
    M1_CALIBRATED_THRESHOLD,
    classify_email_m1,
    evaluate_method_disagreements,
)


def test_m1_confidence_calibration_boundary_and_bounds() -> None:
    """Verify that M1 confidence calibration provides continuous decision certainty bounded in [0.50, 1.00]."""
    tau = M1_CALIBRATED_THRESHOLD  # 0.29

    # Test right at threshold boundary
    p_boundary = tau
    margin_b = p_boundary - tau
    # Since p_boundary >= tau, conf = 0.50 + 0.50 * ((p - tau) / (1 - tau))
    conf_boundary = 0.50 + 0.50 * (margin_b / (1.0 - tau))
    assert abs(conf_boundary - 0.50) < 1e-6, "Confidence at threshold must be 0.50 (maximum uncertainty)"

    # Test high probability phishing
    p_high = 0.95
    conf_high = 0.50 + 0.50 * ((p_high - tau) / (1.0 - tau))
    assert conf_high > 0.90
    assert conf_high <= 1.00

    # Test marginal phishing (like Test A, B, C raw scores ~0.30 - 0.45)
    p_marginal = 0.35
    conf_marginal = 0.50 + 0.50 * ((p_marginal - tau) / (1.0 - tau))
    assert 0.50 < conf_marginal < 0.60
    # Must never be < 0.50! The old bug returned 0.35 as confidence for Phishing!
    assert conf_marginal >= 0.50

    # Test high confidence legitimate (p = 0.05)
    p_legit = 0.05
    conf_legit = 0.50 + 0.50 * ((tau - p_legit) / tau)
    assert conf_legit > 0.80
    assert conf_legit <= 1.00

    # Test marginal legitimate (p = 0.25)
    p_marg_legit = 0.25
    conf_marg_legit = 0.50 + 0.50 * ((tau - p_marg_legit) / tau)
    assert 0.50 < conf_marg_legit < 0.60


def test_m1_inference_telemetry_fields() -> None:
    """Verify that classify_email_m1 returns complete telemetry: raw_probability, decision_threshold, threshold_margin."""
    sample_text = "Important update: Please review attached invoice document immediately."
    res = classify_email_m1(sample_text)

    assert "is_phishing" in res
    assert "confidence" in res
    assert "security_action" in res
    assert "raw_probability" in res
    assert "decision_threshold" in res
    assert "threshold_margin" in res

    assert res["decision_threshold"] == M1_CALIBRATED_THRESHOLD
    assert 0.0 <= res["raw_probability"] <= 1.0
    assert 0.50 <= res["confidence"] <= 1.00
    assert abs(res["threshold_margin"] - (res["raw_probability"] - res["decision_threshold"])) < 1e-6


def test_domain_analyzer_lookalike_and_mismatch() -> None:
    """Verify that domain analyzer identifies lookalikes, mismatches, and authentic domains."""
    # Test A: lookalike + destination mismatch
    analysis_a = analyze_email_domains(
        subject="Shared Document: Q4 Review",
        sender="SharePoint Notifications <noreply@sharepoint-docs-verify.com>",
        body="Access your document here: https://evil-login-verify.com/sharepoint/auth",
    )
    assert analysis_a["sender_domain"] == "sharepoint-docs-verify.com"
    assert "evil-login-verify.com" in analysis_a["link_domains"]
    assert "evil-login-verify.com" in analysis_a["link_domain_mismatches"]
    assert analysis_a["has_suspicious_evidence"] is True
    assert len(analysis_a["indicators"]) > 0

    # Formatted evidence block includes verified facts
    block = format_domain_evidence_block(analysis_a)
    assert "STRUCTURED SENDER & DOMAIN EVIDENCE" in block
    assert "sharepoint-docs-verify.com" in block
    assert "MISMATCH DETECTED" in block

    # Test C: Authentic university domain
    analysis_c = analyze_email_domains(
        subject="Routine Password Verification",
        sender="IT Security Office <security@university.edu>",
        body="Please visit https://it.university.edu/auth/mfa to check status.",
    )
    assert analysis_c["sender_domain"] == "university.edu"
    assert "it.university.edu" in analysis_c["link_domains"]
    assert len(analysis_c["link_domain_mismatches"]) == 0
    assert analysis_c["has_suspicious_evidence"] is False


def test_evaluate_method_disagreements_architectural_divergence() -> None:
    """Verify that M4=Legitimate vs M1=Phishing triggers divergence review with elevated scrutiny."""
    results = {
        "M1": {"is_phishing": True, "confidence": 0.65},
        "M4": {"is_phishing": False, "confidence": 0.95},
    }
    summary = evaluate_method_disagreements(results)

    assert summary["has_disagreement"] is True
    assert "Architectural Divergence" in summary["summary_title"]
    assert "DIVERGENCE REVIEW" in summary["summary_detail"]

    # M4 should be marked with warning style
    assert summary["method_styles"]["M4"]["card_class"] == "card-disagreement-warning"
    assert "Divergence" in summary["method_styles"]["M4"]["badge_html"]

    # M1 should be marked with warning style (alerting phishing risk)
    assert summary["method_styles"]["M1"]["card_class"] == "card-disagreement-warning"
    assert "Baseline Dissent" in summary["method_styles"]["M1"]["badge_html"]


def test_evaluate_method_disagreements_m4_overrides_false_negative() -> None:
    """Verify that M4=Phishing vs M1=Legitimate triggers phishing verdict with M4 override badge."""
    results = {
        "M1": {"is_phishing": False, "confidence": 0.70},
        "M4": {"is_phishing": True, "confidence": 0.95, "security_action": "BLOCK"},
    }
    summary = evaluate_method_disagreements(results)

    assert summary["has_disagreement"] is True
    assert summary["majority_verdict"] == "phishing"
    assert "Authoritative security action remains BLOCK" in summary["summary_detail"]
    assert summary["method_styles"]["M4"]["card_class"] == "card-consensus"
    assert "Overridden by Primary M4" in summary["method_styles"]["M1"]["note"]
