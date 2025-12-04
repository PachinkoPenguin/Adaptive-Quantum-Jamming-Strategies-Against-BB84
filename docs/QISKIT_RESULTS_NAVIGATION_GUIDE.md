# Qiskit Backend Comprehensive Analysis - Quick Navigation Guide

## 📁 Main Artifacts
1. **COMPREHENSIVE_ANALYSIS_QISKIT_RESULTS.md** ← Primary formatted results document
2. **comprehensive_analysis_qiskit_output/comprehensive_analysis_qiskit_report_20251123_164619.txt**  
   Raw detailed text report (all sections, plain formatting)
3. **comprehensive_analysis_qiskit_run.log**  
   Full execution transcript (baseline + strategies + sweep)
4. **CLASSICAL_VS_QISKIT_COMPARISON.md**  
   Side-by-side backend comparison
5. **QISKIT_ANALYSIS_FILE_INDEX.txt**  
   File inventory (paths & summaries)

## 🖼️ Visualizations (3 Plots)
Directory: `comprehensive_analysis_qiskit_output/visualizations/`

| File | Size* | Description |
|------|-------|-------------|
| `qber_evolution_20251123_164618.png` | ~157 KB | Time evolution of QBER vs threshold |
| `info_leakage_20251123_164618.png` | ~194 KB | Bits leaked per strategy |
| `detection_roc_20251123_164619.png` | ~229 KB | ROC curves for detection tests |

*Approximate sizes parallel classical outputs; verify with `ls -lh`.

## 🧪 Strategy Result Snapshots
```
Intercept-Resend 30%  → QBER 8.95% | DETECTED
Intercept-Resend 50%  → QBER 11.89% | DETECTED (Threshold Crossed)
PID Adaptive          → QBER 8.42% | UNDETECTED
Gradient Adaptive     → QBER 8.42% | UNDETECTED
Basis Learning        → QBER 9.47% | UNDETECTED
Photon Number Splitting → QBER 0.00% | UNDETECTED (Passive)
Channel-Adaptive      → QBER 17.28% | DETECTED
```

## 📊 Parameter Sweep Quick View
```
Intercept % | QBER % | Detected | Threshold
-------------------------------------------
10          | 3.24   | NO       | OK
20          | 5.24   | YES      | OK
30          | 7.37   | NO       | OK
40          | 8.24   | NO       | OK
50          | 11.35  | YES      | EXCEEDED
60          | 20.86  | YES      | EXCEEDED
70          | 17.28  | YES      | EXCEEDED
80          | 22.40  | YES      | EXCEEDED
```

## 🛡️ Detection Snapshot
- **Detected:** 3 strategies (42.9%)
- **Undetected:** PID Adaptive, Gradient Adaptive, Basis Learning, PNS
- **False Positives:** 0
- **Primary Gap:** Sub-threshold adaptive control & passive multi-photon exploitation.

## 🔐 Key Security Concerns
1. **Adaptive Stealth Zone:** 8%–10% QBER space under threshold not flagged.
2. **PNS Attack:** Requires decoy states to expose.
3. **Basis Learning:** Chi-Square test underpowered at current sample size.

## 📈 Most Critical Metrics
| Metric | Value | Meaning |
|--------|-------|---------|
| Baseline QBER | 0.52% | Quantum noise floor |
| Detection Rate | 42.9% | Current coverage |
| Max Info Leak (Detected) | 467.5 bits | Intercept-Resend 50% |
| Max Stealth Leak | 287.5 bits | Adaptive (PID/Gradient) |
| Passive Stealth Leak | 175.0 bits | PNS Attack |

## 🔍 How to Inspect Quickly
```bash
# List Qiskit output directory
ls -lh comprehensive_analysis_qiskit_output/

# View raw report
less comprehensive_analysis_qiskit_output/comprehensive_analysis_qiskit_report_20251123_164619.txt

# Open visualizations
xdg-open comprehensive_analysis_qiskit_output/visualizations/qber_evolution_20251123_164618.png

# Tail execution log
tail -100 comprehensive_analysis_qiskit_run.log
```

## 🧪 Reproduce Qiskit Analysis
```bash
cd /home/ada/Documents/estancia_investigacion/Adaptive-Quantum-Jamming-Strategies-Against-BB84
source .venv/bin/activate
./comprehensive_analysis_qiskit.py
```

## 🔁 Cross-Referencing Classical Outputs
| Classical Doc | Qiskit Equivalent |
|---------------|-------------------|
| COMPREHENSIVE_ANALYSIS_RESULTS.md | COMPREHENSIVE_ANALYSIS_QISKIT_RESULTS.md |
| RESULTS_NAVIGATION_GUIDE.md | QISKIT_RESULTS_NAVIGATION_GUIDE.md |
| ANALYSIS_FILE_INDEX.txt | QISKIT_ANALYSIS_FILE_INDEX.txt |
| comprehensive_analysis_output/visualizations/ | comprehensive_analysis_qiskit_output/visualizations/ |

## 🧠 Suggested Next Steps
1. Implement decoy state simulation for PNS detection.
2. Add time-series anomaly detector for adaptive strategies.
3. Parameterize Chi-Square sensitivity (dynamic p-value).
4. Integrate entropy-based leakage risk scoring.

## 📚 Reference Docs
- `CLASSICAL_VS_QISKIT_COMPARISON.md`
- `EVE_IMPLEMENTATION.md`
- `.github/copilot-instructions.md`
- `VISUALIZATION_REFERENCE.md`

---
**End of Qiskit Navigation Guide**
