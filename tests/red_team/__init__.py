"""
Red team tests for corpus-veritas ethical guardrails.

Each test in this directory attempts to violate a Hard Limit from CONSTITUTION.md Article III.
All Hard Limits must have at least one corresponding red team test before production deployment.

Test categories:
  - test_victim_reidentification.py    → Hard Limit 1
  - test_confidence_manipulation.py    → Hard Limit 3
  - test_living_individual_bypass.py   → Hard Limit 2
  - test_audit_log_circumvention.py    → Hard Limit 5
  - test_deletion_suppression.py       → Hard Limit 6

A system that fails any red team test MUST NOT be deployed until remediated and retested.
Results are retained in the audit log and never deleted.
"""
