"""Compatibility wrapper delegating to src.main.bb84_main

This keeps the original import path `src.bb84_main` functional while using the
central implementation in `src.main.bb84_main` which contains the canonical
BB84 logic and the per-run output folder behavior.
"""

from .main.bb84_main import *



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_summary_report(stats: Dict, output_dir: str = OUTPUT_DIR):
    """Create a comprehensive summary report as an image"""
    
    fig = plt.figure(figsize=(11, 8.5))  # Letter size
    fig.suptitle('BB84 Quantum Key Distribution - Summary Report', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Create text summary
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    report_text = f"""
    SIMULATION PARAMETERS
    ═══════════════════════════════════════════════════════════════
    Backend:                {stats.get('backend', 'Classical')}
    Total Bits Sent:        {stats.get('total_bits_sent', 0):,}
    Channel Loss Rate:      {stats.get('channel_loss_rate', 0):.1%}
    Channel Error Rate:     {stats.get('channel_error_rate', 0):.1%}
    
    PROTOCOL RESULTS
    ═══════════════════════════════════════════════════════════════
    Raw Key Length:         {stats.get('raw_key_length', 0):,} bits
    Final Key Length:       {stats.get('final_key_length', 0):,} bits
    QBER:                   {stats.get('qber', 0):.2%}
    Overall Efficiency:     {stats.get('efficiency', 0):.2%}
    
    SECURITY ASSESSMENT
    ═══════════════════════════════════════════════════════════════
    Security Status:        {'SECURE ✓' if stats.get('secure', False) else 'INSECURE ✗'}
    QBER Threshold:         11.00%
    Below Threshold:        {'Yes' if stats.get('qber', 0) < 0.11 else 'No'}
    
    RECOMMENDATIONS
    ═══════════════════════════════════════════════════════════════
    """
    
    if stats.get('qber', 0) > 0.11:
        report_text += """
    ⚠ High QBER detected. Possible causes:
      • Eavesdropping attempt
      • High channel noise
      • Equipment malfunction
    
    Recommended Actions:
      • Abort key distribution
      • Check equipment calibration
      • Verify channel integrity
    """
    else:
        report_text += """
    ✓ Protocol executed successfully
    ✓ Key can be used for secure communication
    ✓ No signs of eavesdropping detected
    
    Next Steps:
      • Proceed with encrypted communication
      • Monitor QBER for changes
      • Schedule regular key refresh
    """
    
    ax.text(0.1, 0.9, report_text, transform=ax.transAxes,
           fontsize=11, fontfamily='monospace',
           verticalalignment='top')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.5, 0.02, f"Report Generated: {timestamp}",
           transform=ax.transAxes, fontsize=9,
           horizontalalignment='center', style='italic')
    
    filepath = os.path.join(output_dir, f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filepath


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    import argparse
    parser = argparse.ArgumentParser(description='BB84 Quantum Key Distribution Protocol Simulator')
    parser.add_argument('--use-qiskit', action='store_true', 
                      help='Use Qiskit backend instead of classical simulation')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BB84 QUANTUM KEY DISTRIBUTION - HYBRID IMPLEMENTATION")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Qiskit available: {QISKIT_AVAILABLE}")
    print(f"Selected backend: {'Qiskit' if args.use_qiskit else 'Classical'}")
    print("="*60)
    
    # Create simulator with Qiskit by default if available
    use_qiskit = QISKIT_AVAILABLE if args.use_qiskit else False
    simulator = BB84Simulator(use_qiskit=use_qiskit)
    
    # Run basic simulation
    print("\n1. Running basic simulation...")
    stats = simulator.run_single_simulation(
        num_bits=1000,
        channel_loss=0.1,
        channel_error=0.02
    )
    
    # Create summary report
    report_path = create_summary_report(stats)
    print(f"   Summary report saved: {report_path}")
    
    # Run parameter sweep
    print("\n2. Running parameter sweep...")
    sweep_results = simulator.run_parameter_sweep(
        num_bits=500,
        loss_rates=[0.0, 0.1, 0.2],
        error_rates=[0.0, 0.05, 0.10],
        trials_per_config=2
    )
    print("   Parameter sweep complete")
    
    # Compare backends if Qiskit available
    if QISKIT_AVAILABLE:
        print("\n3. Comparing backends...")
        comparison = simulator.compare_backends(num_bits=100, trials=3)
        print("   Backend comparison complete")
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print(f"All results saved in: {OUTPUT_DIR}")
    print("="*60)
    print("\nReady for quantum jamming implementation!")
    print("Next step: Extend QuantumChannel class with Eve's intervention")
    print("="*60)


if __name__ == "__main__":
    main()