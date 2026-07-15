# Resolved Anomaly Cases

## Case 001 — null_spike total_amount + payment_method
Store: Carrefour Express Gauthier Proximite Casablanca
Cause: POS terminal WiFi lost during peak hour. Partial record serialization.
Fix: IT restarted WiFi, re-entered 12 transactions from receipts.
Confidence: 0.92

## Case 002 — schema_drift pipeline-wide
Cause: Developer renamed field currency to currency_code in v2.1.0.
Fix: Rolled back producer in 12 minutes. Added schema registry.
Confidence: 0.95

## Case 003 — range_violation total_amount = -2450 MAD
Store: Carrefour Hyper Ain Sebaa
Cause: Return processed through sale screen not return screen.
Fix: Corrected transaction, added PIN for negative amounts.
Confidence: 0.78

## Case 004 — categorical_drift payment_method = wafacash
Cause: Two stores accepting WafaCash in regional pilot without updating config.
Fix: Added wafacash to allowed values in pipelines.yaml.
Confidence: 0.88

## Key pattern
- RAG improves confidence significantly: Ramadan case 0.91 with RAG vs 0.44 without.
- null_spike on order_id or store_id: always critical.
- schema_drift: always escalate and pause pipeline.
