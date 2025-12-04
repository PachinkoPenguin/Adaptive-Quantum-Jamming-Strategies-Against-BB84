# VisualizationManager Quick Reference

## Overview
The `VisualizationManager` class provides comprehensive visualization and analysis tools for BB84 quantum key distribution attack research. It generates publication-quality figures and LaTeX tables suitable for academic papers.

## Initialization

```python
from src.main.bb84_main import VisualizationManager

# Create visualization manager with custom output directory
viz = VisualizationManager(output_dir="my_figures")
```

## Publication Settings

All plots use these publication-quality settings:
- Figure size: 10×6 inches
- DPI: 150 (display), 300 (saved files)
- Font sizes: 12pt (body), 14pt (labels), 16pt (titles)
- Grid and styling optimized for IEEE/ACM papers

---

## Basic Plotting Methods

### 1. QBER Evolution Over Time

```python
qber_history = [0.05, 0.08, 0.11, 0.13, 0.12, ...]  # List of QBER values
filepath = viz.plot_qber_evolution(
    qber_history,
    threshold=0.11,                    # Detection threshold line
    title="QBER Evolution Over Time",
    save_name="my_qber_plot.png"       # Optional custom filename
)
```

**Features:**
- Time series plot with threshold line
- Highlights regions where QBER exceeds threshold (attack detected)
- Automatic y-axis scaling

---

### 2. Intercept Probability Adaptation

```python
intercept_prob_history = [0.8, 0.75, 0.7, 0.6, ...]  # Adaptive behavior
filepath = viz.plot_intercept_probability(
    intercept_prob_history,
    title="Intercept Probability Adaptation"
)
```

**Features:**
- Shows Eve's adaptive strategy over time
- Includes moving average smoothing (if >10 data points)
- Marker points for visibility

---

### 3. Information Leakage Comparison

```python
strategies = {
    'Intercept-Resend': 0.50,
    'QBER-Adaptive': 0.35,
    'Basis-Learning': 0.42,
    'PNS Attack': 0.28
}
filepath = viz.plot_information_leakage(
    strategies,
    title="Information Leakage Comparison"
)
```

**Features:**
- Bar chart with color gradient
- Value labels on each bar
- Ideal for strategy comparisons

---

### 4. Detection ROC Curves

```python
roc_data = {
    'QBER Detection': {
        'fpr': [0.0, 0.1, 0.2, ..., 1.0],
        'tpr': [0.0, 0.6, 0.8, ..., 1.0]
    },
    'Chi-Square Test': {
        'fpr': [...],
        'tpr': [...]
    }
}
filepath = viz.plot_detection_roc_curve(
    roc_data,
    title="Detection ROC Curves"
)
```

**Features:**
- Multiple ROC curves on same plot
- Automatic AUC (Area Under Curve) calculation
- Reference diagonal for random classifier
- Square aspect ratio (1:1)

---

## Advanced Dashboards

### Adaptive Strategy Evolution Dashboard

Comprehensive 4-subplot figure showing:
1. QBER evolution with confidence bands
2. Intercept probability adaptation
3. Cumulative information gain
4. Detection metrics evolution

```python
filepath = viz.plot_adaptive_strategy_dashboard(
    qber_history=[0.05, 0.08, ...],
    intercept_prob_history=[0.8, 0.7, ...],
    info_gain_history=[0.0, 0.01, 0.03, ...],  # Cumulative
    detection_metrics={
        'QBER_exceeded': [0, 0, 1, 1, ...],
        'Chi_Square_p': [0.8, 0.6, 0.3, ...],
        'MI_detected': [0, 0, 0, 1, ...]
    },
    qber_threshold=0.11
)
```

**Output:** Large 16×12 inch figure with 2×2 subplot grid

---

## Atmospheric Effects

### 1. Rytov Variance (Turbulence Strength)

```python
time = [0, 0.1, 0.2, ..., 100.0]  # Time in seconds
rytov_values = [0.1, 0.15, 0.8, ...]  # Rytov variance σ²ᴿ

filepath = viz.plot_rytov_variance(time, rytov_values)
```

**Features:**
- Log scale y-axis for wide dynamic range
- Threshold lines for weak/strong turbulence regimes
- Filled area under curve

---

### 2. Attack Aggression vs Turbulence

```python
turbulence_levels = [0.1, 0.3, 0.8, ...]  # Rytov variance or cn²
aggression_levels = [0.9, 0.7, 0.5, ...]  # Intercept probabilities
qber_values = [0.05, 0.12, 0.18, ...]     # For color coding

filepath = viz.plot_attack_aggression_vs_turbulence(
    turbulence_levels,
    aggression_levels,
    qber_values
)
```

**Features:**
- Scatter plot with QBER color map
- Shows relationship between environmental conditions and attack strategy
- Useful for channel-adaptive attack analysis

---

### 3. Atmospheric Phase Screen

```python
import numpy as np

# Generate or load phase screen
phase_screen = np.random.randn(128, 128) * 2 * np.pi

filepath = viz.plot_phase_screen(phase_screen)
```

**Features:**
- 2D heatmap with "twilight" colormap
- Phase values in radians
- Bilinear interpolation for smooth visualization

---

### 4. Beam Propagation Animation

```python
# Generate sequence of intensity frames
frames = []
for i in range(20):
    intensity = generate_beam_intensity(step=i)  # Your function
    frames.append(intensity)

frame_dir = viz.animate_beam_propagation(
    frames,
    save_name="beam_animation",
    fps=10
)
```

**Output:** Directory with individual PNG frames + metadata.json

**Note:** To create actual video, use external tools like `ffmpeg`:
```bash
ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 beam_propagation.mp4
```

---

## Statistical Analysis

### 1. Error Pattern Analysis

Analyzes error sequences with:
- Error visualization
- Autocorrelation function
- Run length distribution

```python
errors = [0, 0, 1, 0, 1, 1, 0, ...]  # Binary: 0=correct, 1=error

filepath = viz.plot_error_pattern_analysis(errors)
```

**Use cases:**
- Detect non-random error patterns
- Identify error clustering (sign of eavesdropping)
- Runs test visualization

---

### 2. Basis Bias Detection

Chi-square test for basis randomness:

```python
basis_sequence = [0, 1, 0, 0, 1, ...]  # 0 or 1 for each qubit

filepath = viz.plot_basis_bias_detection(
    basis_sequence,
    expected_prob=0.5  # Expected probability for random basis choice
)
```

**Features:**
- Cumulative proportion plot with confidence intervals
- Chi-square statistic over sliding windows
- Critical value threshold line

---

### 3. Mutual Information Matrix

Visualize information sharing between Alice, Bob, and Eve:

```python
mi_matrix = np.array([
    [1.000, 0.920, 0.085],  # Alice
    [0.920, 1.000, 0.072],  # Bob
    [0.085, 0.072, 1.000]   # Eve
])

filepath = viz.plot_mutual_information_matrix(
    mi_matrix,
    labels=['Alice', 'Bob', 'Eve']
)
```

**Features:**
- Heatmap with annotated values
- Symmetric matrix visualization
- Color gradient shows information correlation

---

### 4. Holevo Bound Comparison

Compare theoretical bounds with actual measurements:

```python
strategies = ['Intercept\nResend', 'QBER\nAdaptive', 'PNS']
theoretical_bounds = [1.0, 0.85, 0.75]  # Holevo bounds (bits)
actual_information = [0.50, 0.35, 0.28]  # Measured info gain

filepath = viz.plot_holevo_bound_comparison(
    strategies,
    theoretical_bounds,
    actual_information
)
```

**Features:**
- Side-by-side bar comparison
- Value labels for precise reading
- Useful for validating information-theoretic calculations

---

## LaTeX Table Export

### Publication-Ready Results Tables

Generate IEEE two-column format LaTeX tables:

```python
results = {
    'Strategy_A': {
        'QBER': {'mean': 0.108, 'std': 0.005, 'significant': True},
        'Key_Rate': {'mean': 218.7, 'std': 11.3, 'significant': False},
        'Detection': {'mean': 0.456, 'std': 0.082, 'significant': True}
    },
    'Strategy_B': {
        'QBER': {'mean': 0.095, 'std': 0.007, 'significant': False},
        'Key_Rate': {'mean': 235.1, 'std': 9.8, 'significant': False},
        'Detection': {'mean': 0.382, 'std': 0.091, 'significant': False}
    }
}

filepath = viz.export_results_table(
    results,
    filename="my_results.tex",
    caption="Experimental Results Summary"
)
```

**Output LaTeX format:**
```latex
\begin{table}[t]
\centering
\caption{Experimental Results Summary}
\label{tab:results}
\begin{tabular}{lccc}
\hline
Strategy & QBER & Key_Rate & Detection \\
\hline
Strategy_A & $0.108 \pm 0.005$$^*$ & $218.700 \pm 11.300$ & $0.456 \pm 0.082$$^*$ \\
Strategy_B & $0.095 \pm 0.007$ & $235.100 \pm 9.800$ & $0.382 \pm 0.091$ \\
\hline
\end{tabular}
\vspace{2mm}
\\{\footnotesize $^*$ indicates statistical significance at $p < 0.05$}
\end{table}
```

**Features:**
- Automatic mean ± std formatting
- Statistical significance markers ($^*$)
- Proper LaTeX escaping for underscores
- IEEE format compliance

---

## Integration with Simulations

### Example: Visualizing Simulation Results

```python
from src.main.bb84_main import (
    VisualizationManager,
    AdaptiveJammingSimulator,
    InterceptResendAttack,
    ClassicalBackend
)

# Run simulation
backend = ClassicalBackend()
attack = InterceptResendAttack(backend=backend, intercept_probability=0.5)
sim = AdaptiveJammingSimulator(
    n_qubits=1000,
    attack_strategy=attack,
    backend=backend
)
results = sim.run_simulation()

# Visualize results
viz = VisualizationManager(output_dir="simulation_results")

# Create information leakage plot
if 'eve_info' in results:
    info_dict = {'Intercept-Resend': results['eve_info']}
    viz.plot_information_leakage(info_dict)

# Create QBER plot if history available
if 'qber_history' in results:
    viz.plot_qber_evolution(results['qber_history'])
```

---

## Tips and Best Practices

### 1. File Organization
```python
# Organize outputs by experiment
viz_exp1 = VisualizationManager("results/experiment_1")
viz_exp2 = VisualizationManager("results/experiment_2")
```

### 2. Consistent Naming
```python
# Use descriptive filenames
viz.plot_qber_evolution(data, save_name="exp1_qber_n1000_loss010.png")
```

### 3. Batch Processing
```python
# Process multiple datasets
for experiment in experiments:
    filepath = viz.plot_qber_evolution(
        experiment['qber'],
        save_name=f"qber_{experiment['name']}.png"
    )
    print(f"Saved: {filepath}")
```

### 4. Custom Styling
```python
import matplotlib.pyplot as plt

# Modify rcParams before creating VisualizationManager
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.facecolor'] = 'white'

viz = VisualizationManager()
```

---

## Testing

Run visualization tests:
```bash
pytest tests/test_visualization.py -v
```

Run demonstration script:
```bash
python examples/visualization_demo.py
```

---

## Troubleshooting

### Issue: Plots look pixelated
**Solution:** Increase DPI in saved files
```python
plt.rcParams['savefig.dpi'] = 600  # Very high resolution
```

### Issue: LaTeX symbols not rendering
**Solution:** Install LaTeX backend for matplotlib
```bash
sudo apt-get install cm-super dvipng
```

### Issue: Memory error with large animations
**Solution:** Process frames in batches
```python
# Save frames incrementally instead of storing all in memory
for i, frame in enumerate(generate_frames()):
    viz.animate_beam_propagation([frame], save_name=f"batch_{i}")
```

---

## References

- **Matplotlib Documentation**: https://matplotlib.org/
- **LaTeX Table Guide**: https://www.overleaf.com/learn/latex/Tables
- **IEEE Paper Format**: https://www.ieee.org/publications/authors

---

## Summary of Methods

| Method | Purpose | Key Features |
|--------|---------|--------------|
| `plot_qber_evolution` | QBER time series | Threshold line, detection regions |
| `plot_intercept_probability` | Adaptive parameter | Moving average, trend analysis |
| `plot_information_leakage` | Strategy comparison | Bar chart, value labels |
| `plot_detection_roc_curve` | Detector performance | AUC calculation, multiple curves |
| `plot_adaptive_strategy_dashboard` | Comprehensive view | 4 subplots, confidence bands |
| `plot_rytov_variance` | Turbulence evolution | Log scale, regime thresholds |
| `plot_attack_aggression_vs_turbulence` | Environmental correlation | Scatter plot, QBER coloring |
| `plot_phase_screen` | Atmospheric phase | 2D heatmap, phase visualization |
| `animate_beam_propagation` | Temporal dynamics | Frame sequence, metadata |
| `plot_error_pattern_analysis` | Error statistics | Autocorrelation, runs, distribution |
| `plot_basis_bias_detection` | Randomness testing | Chi-square, confidence intervals |
| `plot_mutual_information_matrix` | Information flow | Heatmap, party relationships |
| `plot_holevo_bound_comparison` | Theoretical validation | Bound vs actual comparison |
| `export_results_table` | Paper tables | LaTeX format, significance markers |

---

**For full examples, see:** `examples/visualization_demo.py`
