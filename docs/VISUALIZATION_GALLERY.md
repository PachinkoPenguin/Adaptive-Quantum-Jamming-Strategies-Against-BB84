# Visualization Gallery

This gallery showcases the kinds of figures you can generate with `VisualizationManager` and the end-to-end demo in `examples/visualization_demo.py`.

Use the reference in `VISUALIZATION_REFERENCE.md` for API-level details and `examples/visualization_demo.py` to regenerate a comprehensive set.

---

## Representative visualizations

### Adaptive strategy dashboard
Four subplots presenting: QBER with confidence bands, intercept probability adaptation, cumulative information gain, and detection metrics over time. Useful for summarizing a strategy’s behavior and detectability.

### Information leakage comparison
Bar chart comparing estimated or measured information leakage across strategies (e.g., Intercept-Resend, QBER-Adaptive, PNS). Great for high-level comparisons in papers.

### Detection ROC curves
Multiple ROC curves (QBER-only vs combined detectors) with AUC. Shows trade-offs across detection thresholds.

### QBER evolution
Time series with an 11% threshold line and shaded detection regions; ideal for attack onset/response narratives.

---

## Atmospheric effects (examples)

### Rytov variance over time or distance
Explores turbulence regimes and their masking effect on QBER and detection. Includes threshold/reference lines and optional wavelength overlays.

### Phase screens and beam propagation
2D phase heatmaps and frame sequences of beam intensity propagating through turbulence; useful for illustrating atmospheric impact.

---

## Locations

All visualizations created by the protocol, simulator, or demo are saved within the corresponding run directory under `bb84_output/` (or a directory you pass to `VisualizationManager`).

---

## Regenerating a figure set

Run the example demo:
```bash
python examples/visualization_demo.py
```
Or call individual methods from `VisualizationManager` directly (see `VISUALIZATION_REFERENCE.md`).

---

## Visualization techniques

### Plotting Libraries

- **Matplotlib**: All visualizations use matplotlib
- **Figure sizes**: 14-16 inches wide, 10-12 inches tall
- **DPI**: 100 (high quality)
- **Style**: Clean, academic presentation

### Color Schemes

- Clear categorical palettes for strategy comparisons and detection flags
- Sequential/continuous colormaps for heatmaps and phase screens (e.g., viridis, twilight)

### Plot Types

1. **Line plots**: Time series, convergence tracking
2. **Bar plots**: Discrete metrics, regime classification
3. **Histograms**: Particle distributions, weights
4. **Heatmaps**: Phase screens, correlation matrices
5. **Scatter plots**: Particle locations, phase space
6. **Step functions**: Regime-based policies

---

## Visual insights (typical)

- QBER often reveals attack onset; combined detectors reduce false negatives.
- PID-based strategies track thresholds tightly and can remain below 11% QBER.
- Atmospheric turbulence can camouflage attacks; phase and Rytov plots explain why.

---

## Publication-Ready Quality

All visualizations are:
- ✅ High resolution (100 DPI)
- ✅ Clear labeling (axis labels, titles, legends)
- ✅ Appropriate scales (linear/log as needed)
- ✅ Color-blind friendly (where possible)
- ✅ Academic style (clean, professional)
- ✅ Self-contained (understandable without text)

### Recommended Uses

1. **Research Papers**: Figures 5-10 suitable for journal articles
2. **Presentations**: All figures work well in slides
3. **Thesis/Dissertation**: Complete figure set for chapters
4. **Educational Material**: Clear enough for teaching
5. **Documentation**: Integrated into markdown docs

---

## Customization

All visualizations can be customized by modifying the demo files:

Adjust parameters in your own scripts or in `examples/visualization_demo.py` (e.g., figure sizes, data smoothing, thresholds). All methods return file paths so you can integrate them into pipelines.

---

## Conclusion

The visualization stack produces publication-ready figures covering protocol behavior, attack dynamics, detection performance, atmospheric effects, and information-theoretic comparisons. Use the reference for method-level details and the demo to reproduce a complete figure set.
