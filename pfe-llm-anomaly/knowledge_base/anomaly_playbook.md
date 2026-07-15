# Anomaly Playbook

## null_spike
Causes: POS terminal network failure, Kafka producer crash, schema change.
Store fix: Restart POS terminal, call IT ext. 4400.
Engineer fix: Check producer logs, verify EXPECTED_SCHEMA.
Severity: high if > 10% null rate

## schema_drift
Causes: New producer version renamed fields, store system update.
Store fix: Contact IT coordinator immediately.
Engineer fix: git log --oneline -10 on producer, update EXPECTED_SCHEMA.
Severity: critical — stop pipeline

## range_violation
Causes: Refund processed as sale (negative total), unit conversion error.
Store fix: Use refund screen for returns.
Engineer fix: Add hard validation at producer.
Severity: medium

## categorical_drift
Causes: New payment method, status changed by system update.
Store fix: Contact regional coordinator.
Engineer fix: Update CATEGORICAL_FIELDS in config.
Severity: low to medium
