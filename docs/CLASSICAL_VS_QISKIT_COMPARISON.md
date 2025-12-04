# BB84 Classical vs Qiskit Backend Comparison
**Generated:** November 23, 2025  
**Scope:** Side-by-side analysis of protocol, detection, and attack behavior across simulation backends.

---

## \ud83d\udcca Baseline Metrics

| Metric | Classical | Qiskit | Delta | Interpretation |
|--------|-----------|--------|-------|----------------|
| Baseline QBER | 1.05% | 0.52% | -0.53% | Quantum backend lower intrinsic noise |
| Final Key Length | 380 bits | 383 bits | +3 | Slight variance due to stochastic effects |
| Efficiency | 19.00% | 19.15% | +0.15% | Marginal improvement from reduced error correction load |
| Attack Detected (Baseline) | False | False | = | No false positives both backends |

**Insight:** Lower Qiskit baseline widens stealth margin for adaptive attacks by ~0.5% absolute QBER.

---

## \ud83c\udf9f\ufe0f Detection System Effectiveness

| Metric | Classical | Qiskit | Delta |
|--------|----------|--------|-------|
| Strategies Tested | 7 | 7 | = |
| Detected | 6 (85.7%) | 3 (42.9%) | -42.8% |
| Undetected | PNS only | PID, Gradient, Basis, PNS | +3 new stealth vectors |
| False Positives | 0 | 0 | = |
| Threshold Crossers (\u226511%) | 3 | 3 | = |

**Interpretation:** Multi-test suite tuned to classical variance misses sub-threshold adaptive strategies under quantum noise.

### Undetected Strategy Set Expansion
- **Classical:** Only passive PNS evades (expected behavior without decoy states).
- **Qiskit:** Active adaptive control + learning strategies evade due to lower baseline + noise dispersion.

---

## \ud83d\udd2c Strategy-Level Comparison

| Strategy | Classical QBER | Classical Detected | Qiskit QBER | Qiskit Detected | Notes |
|----------|----------------|--------------------|-------------|-----------------|-------|
| Intercept-Resend 30% | 9.90% | Yes | 8.95% | Yes | Slightly lower QBER quantum; still detected |
| Intercept-Resend 50% | 18.78% | Yes | 11.89% | Yes | Quantum lowers escalation; threshold still crossed |
| PID Adaptive | 9.90% | Yes | 8.42% | No | Sub-threshold stealth success under quantum |
| Gradient Adaptive | 9.90% | Yes | 8.42% | No | Same pattern as PID |
| Basis Learning | 11.54% | Yes (threshold + stats) | 9.47% | No | Falls below threshold; statistical tests less sensitive |
| Photon Number Splitting | 1.05% | No | 0.00% | No | Passive; quantum sim produces perfect stealth |
| Channel-Adaptive | 17.44% | Yes | 17.28% | Yes | High disturbance detected in both |

**Key Shift:** Adaptive and learning strategies drop below detection surface under quantum simulation.

---

## \ud83d\udccf Information Leakage Comparison

| Strategy | Classical Leak (bits) | Qiskit Leak (bits) | Change | Detected in Classical | Detected in Qiskit |
|----------|-----------------------|--------------------|--------|-----------------------|--------------------|
| Intercept-Resend 50% | 478.0 | 467.5 | -10.5 | Yes | Yes |
| Intercept-Resend 30% | 279.5 | 287.5 | +8.0 | Yes | Yes |
| PID Adaptive | 279.5 | 287.5 | +8.0 | Yes | No |
| Gradient Adaptive | 279.5 | 287.5 | +8.0 | Yes | No |
| Basis Learning | 0.0 | 0.0 | = | Yes | No (Recon only) |
| PNS Attack | 175.0 | 175.0 | = | No | No |
| Channel-Adaptive | 0.0 | 0.0 | = | Yes | Yes |

**Risk Elevation:** Previously detected adaptive strategies now exfiltrate ~287 bits undetected.

---

## \ud83d\udd0d Parameter Sweep Behavior

| Intercept % | Classical QBER % | Detected (Classical) | Qiskit QBER % | Detected (Qiskit) | Observation |
|-------------|------------------|----------------------|---------------|-------------------|-------------|
| 10% | 3.80% | No | 3.24% | No | Both below detection surface |
| 20% | 5.26% | No | 5.24% | Yes | Early detection in Qiskit (sensitivity anomaly) |
| 30% | 9.90% | Yes | 7.37% | No | Classical flags; Qiskit remains sub-threshold |
| 40% | 13.61% | No (threshold exceeded detection gap) | 8.24% | No | Classical threshold exceed but multi-test gap; Qiskit still low |
| 50% | 18.78% | Yes | 11.35% | Yes | Threshold crossing both |
| 60% | 16.38% | Yes | 20.86% | Yes | Quantum overshoot at 60% |
| 70% | 17.44% | Yes | 17.28% | Yes | Comparable disturbance |
| 80% | 23.08% | Yes | 22.40% | Yes | High-intercept convergence |

**Key Divergence:** Qiskit detects at 20% but misses 30%; classical misses 40% despite threshold breach (multi-test configuration nuance).

---

## \ud83d\udee0\ufe0f Root Causes of Detection Divergence
1. **Baseline Shift:** Lower Qiskit baseline reduces statistical separation for adaptive attacks.
2. **Variance Profiles:** Quantum measurement noise increases benign variance, diluting anomaly scores.
3. **Chi-Square Sensitivity:** Basis Learning below threshold requires larger sample or stricter p-value.
4. **Temporal Modeling Absence:** No time-series anomaly detection; adaptive strategies adjust slowly.

---

## \ud83d\udca1 Mitigation Recommendations
| Priority | Action | Rationale |
|----------|--------|-----------|
| High | Implement decoy state protocol | Detect passive PNS exploitation |
| High | Add sub-threshold anomaly detector (trend + entropy deviation) | Catch PID/Gradient stealth |
| Medium | Increase basis batch size & refine Chi-Square thresholds | Improve Basis Learning detection |
| Medium | Introduce adaptive dynamic QBER baseline estimation | Separate natural quantum drift from attack |
| Low | Leak-aware risk scoring (bits vs QBER) | Prioritize investigations |

---

## \ud83d\udcdd Summary Statement
The Qiskit backend exposes a broader stealth surface for adaptive and learning eavesdropping strategies due to lower baseline error and increased benign variance. While BB84 remains fundamentally secure (detected high-disturbance attacks, no false positives), refinements are required to restore classical-level detection coverage in quantum-realistic conditions.

---

**End of Classical vs Qiskit Comparison**
