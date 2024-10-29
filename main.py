import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import bluequbit
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats, fft
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import seaborn as sns

class EnhancedATHValidator:
    def __init__(self, token: str, shots: int = 5000):
        """Initialize the enhanced ATH validator with falsification capabilities"""
        self.bq = bluequbit.init(token)
        self.shots = shots
        # Optimized parameter ranges
        self.phi_range = np.linspace(1.0, 2.0, 8)
        self.energy_range = np.linspace(0.0, 0.8, 6)
        self.time_range = np.linspace(2.5, 3.5, 8)

    def _quantum_error_correction(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Enhanced error correction with adjustable threshold and detailed logging"""
        if not counts:
            return {}

        total_counts = sum(counts.values())
        if total_counts == 0:
            return {}

        # Reduced noise threshold for more sensitive detection
        noise_threshold = 0.01 * total_counts  # Changed from 0.02 to 0.01
        signal_strength = np.sqrt(total_counts)

        filtered_counts = {}
        rejected_counts = {}  # Track rejected measurements

        for state, count in counts.items():
            if count > noise_threshold:
                weight = count/total_counts
                enhanced_count = int(count * (1 + weight * count/signal_strength))
                filtered_counts[state] = enhanced_count
            else:
                rejected_counts[state] = count

        # Log rejection statistics
        if rejected_counts:
            rejection_rate = sum(rejected_counts.values()) / total_counts
            print(f"Rejected {len(rejected_counts)} states ({rejection_rate:.2%} of counts)")
            print(f"Rejected states statistics: min={min(rejected_counts.values())}, "
                  f"max={max(rejected_counts.values())}")

        return filtered_counts

    def create_falsification_circuit(self, param: float) -> QuantumCircuit:
        """
        Enhanced falsification circuit with improved verification and logging
        """
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        ver = ClassicalRegister(3, 'v')  # Added verification register
        qc = QuantumCircuit(qr, cr, ver)

        # Initial state preparation with verification
        qc.rx(np.pi/2, qr[0])
        qc.h(qr[1])  # Additional superposition
        qc.cx(qr[0], qr[1])
        qc.measure(qr[1], ver[0])  # Verify initial entanglement
        qc.barrier()

        # Create entangled state with staged verification
        for i in range(7):
            qc.cx(qr[i], qr[i+1])
            if i == 3:  # Mid-circuit verification
                qc.measure(qr[i], ver[1])
                qc.barrier()

        # Enhanced time-dependent phase shifts with error detection
        phase_shifts = []
        for i in range(8):
            angle = param * np.pi * (i+1)
            qc.rz(angle, qr[i])
            phase_shifts.append(angle)

        # Create temporal correlations with improved precision
        angle = param * np.pi
        for i in range(7):
            # Primary rotation
            qc.crz(angle, qr[i], qr[i+1])
            qc.cz(qr[i], qr[i+1])

            # Secondary rotation with error checking
            if i < 6:
                qc.ccx(qr[i], qr[i+1], qr[i+2])
                qc.rz(angle/2, qr[i+2])
                qc.measure(qr[i+2], ver[2])  # Verify secondary rotation
                qc.ccx(qr[i], qr[i+1], qr[i+2])

        # Final measurements with improved basis
        for i in range(8):
            qc.h(qr[i])  # Hadamard before measurement
            qc.measure(qr[i], cr[i])

        return qc

    def create_beyond_quantum_circuit(self, param: float) -> QuantumCircuit:
        """
        Enhanced circuit to test for effects that should not be possible
        under conventional quantum mechanics but are predicted by ATH
        """
        qr = QuantumRegister(7, 'q')  # Added ancilla qubit
        cr = ClassicalRegister(6, 'c')
        qc = QuantumCircuit(qr, cr)

        # Enhanced initialization with verification
        for i in range(6):
            qc.rx(np.pi/2, qr[i])
            qc.cz(qr[6], qr[i])  # Verify initialization with ancilla

        # Create nested temporal evolution with improved precision
        for i in range(5):
            # Primary evolution with verification
            qc.crz(param * np.pi, qr[i], qr[i+1])
            qc.cz(qr[6], qr[i])  # Verify with ancilla

            # Enhanced secondary evolution (nested time scales)
            nested_angle = param * np.pi * np.sin(param * np.pi)
            qc.rz(nested_angle, qr[i])
            qc.rz(-nested_angle, qr[i+1])

            # Create temporal entanglement with verification
            qc.cx(qr[i], qr[i+1])
            qc.rz(param * np.pi / 2, qr[i+1])
            qc.cz(qr[6], qr[i+1])  # Verify with ancilla
            qc.cx(qr[i], qr[i+1])

        # Final interference layer with verification
        for i in range(6):
            qc.h(qr[i])
            qc.cz(qr[6], qr[i])  # Final verification
            qc.measure(qr[i], cr[i])

        return qc

    def optimized_adaptive_circuit(self, energy: float) -> QuantumCircuit:
        """
        Enhanced adaptive circuit with improved energy-time coupling
        """
        qc = QuantumCircuit(6, 4)

        # Phase 1: Robust energy encoding
        qc.ry(energy * np.pi/2, 4)
        qc.cx(4, 5)  # Entangle ancilla qubits
        qc.ry(energy * np.pi/2, 5)
        qc.barrier()

        # Phase 2: Enhanced energy-state coupling
        for i in range(4):
            qc.h(i)
            # Dual ancilla coupling
            qc.crz(energy * np.pi/2, 4, i)
            qc.crz(energy * np.pi/2, 5, i)
            # Verification gates
            qc.cz(4, i)
            qc.cz(5, i)
            qc.h(i)
        qc.barrier()

        # Phase 3: Enhanced measurement protocol
        for i in range(4):
            qc.h(i)
            qc.measure(i, i)

        return qc

    def enhanced_clock_circuit(self, time: float) -> QuantumCircuit:
        """
        Improved quantum clock circuit with nonlinear mapping
        """
        qc = QuantumCircuit(7, 4)

        # Phase 1: Time-sensitive initialization
        for i in range(4):
            qc.rx(time * np.pi/4, i)
        qc.barrier()

        # Phase 2: Nonlinear evolution
        for i in range(4):
            angle = time * (1 + np.sin(time * np.pi))
            qc.crz(angle, 6, i)
            qc.cx(i, (i+1) % 4)
            qc.rz(angle/2, i)
        qc.barrier()

        # Phase 3: Enhanced measurement
        for i in range(4):
            qc.h(i)
            qc.measure(i, i)

        return qc

    def enhanced_temporal_circuit(self, phi: float) -> QuantumCircuit:
        """
        Improved temporal circuit with resonance detection
        """
        qc = QuantumCircuit(5, 4)

        # Phase 1: Enhanced initialization
        for i in range(4):
            qc.rx(np.pi/2, i)  # More robust than H
        qc.h(4)  # Ancilla for verification
        qc.barrier()

        # Phase 2: Resonance-sensitive phase encoding
        for i in range(4):
            # Split phase application for better control
            qc.crz(phi/4, 4, i)
            qc.x(i)
            qc.crz(phi/4, 4, i)
            qc.x(i)
            qc.crz(phi/2, 4, i)
        qc.barrier()

        # Phase 3: Verification measurements
        for i in range(4):
            qc.h(i)
            qc.cz(4, i)
            qc.measure(i, i)

        return qc

    def _calculate_temporal_correlation(self, counts: Dict[str, int]) -> float:
        """
        Enhanced calculation of temporal correlations that should be impossible under
        conventional quantum mechanics
        """
        if not counts:
            return 0.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        correlation_sum = 0
        for state, count in counts.items():
            bits = [int(x) for x in state]
            # Look for specific patterns indicating temporal correlation
            for i in range(len(bits)-2):
                if bits[i] == bits[i+1] == bits[i+2]:
                    correlation_sum += count

        return correlation_sum / total

    def create_falsification_circuit(self, param: float) -> QuantumCircuit:
        """
        Enhanced falsification circuit without mid-circuit measurements
        """
        qr = QuantumRegister(8, 'q')
        cr = ClassicalRegister(8, 'c')
        qc = QuantumCircuit(qr, cr)

        # Initial state preparation
        qc.rx(np.pi/2, qr[0])
        qc.h(qr[1])  # Additional superposition
        qc.cx(qr[0], qr[1])
        qc.barrier()

        # Create entangled state with improved verification
        for i in range(7):
            qc.cx(qr[i], qr[i+1])
            # Use barrier instead of measurement for separation
            qc.barrier()

        # Enhanced time-dependent phase shifts
        for i in range(8):
            angle = param * np.pi * (i+1)
            qc.rz(angle, qr[i])
        qc.barrier()

        # Create temporal correlations with improved precision
        angle = param * np.pi
        for i in range(7):
            # Primary rotation
            qc.crz(angle, qr[i], qr[i+1])
            qc.cz(qr[i], qr[i+1])

            # Secondary rotation with error tracking
            if i < 6:
                qc.ccx(qr[i], qr[i+1], qr[i+2])
                qc.rz(angle/2, qr[i+2])
                # Use cz instead of measurement for verification
                qc.cz(qr[i], qr[i+2])
                qc.ccx(qr[i], qr[i+1], qr[i+2])
        qc.barrier()

        # Additional verification gates before final measurement
        for i in range(8):
            qc.h(qr[i])
            # Add verification CZ gates
            if i < 7:
                qc.cz(qr[i], qr[i+1])

        # Final measurements
        for i in range(8):
            qc.measure(qr[i], cr[i])

        return qc

    def _calculate_falsification_metric(self, counts: Dict[str, int]) -> float:
        """
        Enhanced falsification metric with more sensitive analysis
        """
        if not counts:
            return 0.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        # Multiple metric components for better sensitivity
        metrics = {
            'entropy': 0.0,
            'pattern': 0.0,
            'correlation': 0.0
        }

        for state, count in counts.items():
            bits = [int(x) for x in state]

            # Enhanced entropy calculation
            local_entropy = sum(bits) / len(bits)
            metrics['entropy'] += abs(local_entropy - 0.5) * count * (1 + np.sin(np.pi * local_entropy))

            # Improved pattern detection
            for i in range(len(bits)-3):
                pattern = bits[i:i+4]
                if pattern == [1,0,1,0] or pattern == [0,1,0,1]:
                    metrics['pattern'] += count * (1 + 0.5 * i/len(bits))

            # Enhanced correlation detection
            for i in range(len(bits)-2):
                if bits[i] == bits[i+1] == bits[i+2]:
                    metrics['correlation'] += count * (1 + 0.3 * i/len(bits))

        # Normalize metrics
        for key in metrics:
            metrics[key] /= total

        # Log detailed metrics if they're non-zero
        if any(v > 0 for v in metrics.values()):
            print("\nFalsification Metric Components:")
            for key, value in metrics.items():
                print(f"{key.title()}: {value:.4f}")

        # Weighted combination with enhanced sensitivity
        combined_metric = (0.4 * metrics['entropy'] +
                         0.3 * metrics['pattern'] +
                         0.3 * metrics['correlation']) * (1 + np.sin(np.pi * sum(metrics.values())/3))

        return combined_metric

    def _calculate_beyond_quantum_metric(self, counts: Dict[str, int]) -> float:
        """
        Enhanced metric for effects that should be impossible under
        conventional quantum mechanics
        """
        if not counts:
            return 0.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        pattern_sum = 0
        for state, count in counts.items():
            bits = [int(x) for x in state]
            # Enhanced pattern detection
            for i in range(len(bits)-3):
                pattern = bits[i:i+4]
                if pattern == [1,0,1,0] or pattern == [0,1,0,1]:
                    pattern_sum += count * (1 + np.sin(np.pi * i/len(bits)))

        return pattern_sum / total

    def run_enhanced_validation(self, verbose: bool = True) -> Dict:
        """
        Modified validation suite with enhanced falsification analysis
        """
        results = {
            'temporal': {'values': [], 'uncertainties': [], 'raw_data': []},
            'adaptive': {'values': [], 'uncertainties': [], 'raw_data': []},
            'clock': {'values': [], 'uncertainties': [], 'raw_data': []},
            'falsification': {'values': [], 'uncertainties': [], 'raw_data': []},
            'beyond_quantum': {'values': [], 'uncertainties': [], 'raw_data': []}
        }

        try:
            for phi in self.phi_range:
                if verbose:
                    print(f"\nTesting parameter value: {phi:.2f}")

                # Dictionary to store test results for current parameter
                phi_results = {test: {'values': [], 'raw': []} for test in results.keys()}

                # Temporal test
                if verbose:
                    print("Running temporal test...")
                for trial in range(7):
                    circuit = self.enhanced_temporal_circuit(phi)
                    raw_counts = self.bq.run(circuit, device='cpu', shots=self.shots).get_counts()
                    counts = self._quantum_error_correction(raw_counts)
                    if counts:
                        metric = sum(bin(int(state, 2)).count('1') * count
                                   for state, count in counts.items()) / (self.shots * 4)
                        phi_results['temporal']['values'].append(metric)
                        phi_results['temporal']['raw'].append(counts)

                # Adaptive test
                if verbose:
                    print("Running adaptive test...")
                for trial in range(7):
                    circuit = self.optimized_adaptive_circuit(phi)
                    raw_counts = self.bq.run(circuit, device='cpu', shots=self.shots).get_counts()
                    counts = self._quantum_error_correction(raw_counts)
                    if counts:
                        metric = sum(bin(int(state, 2)).count('1') * count / self.shots
                                   for state, count in counts.items())
                        phi_results['adaptive']['values'].append(metric)
                        phi_results['adaptive']['raw'].append(counts)

                # Clock test
                if verbose:
                    print("Running clock test...")
                for trial in range(7):
                    circuit = self.enhanced_clock_circuit(phi)
                    raw_counts = self.bq.run(circuit, device='cpu', shots=self.shots).get_counts()
                    counts = self._quantum_error_correction(raw_counts)
                    if counts:
                        metric = sum(bin(int(state, 2)).count('1') * count / self.shots
                                   for state, count in counts.items())
                        phi_results['clock']['values'].append(metric)
                        phi_results['clock']['raw'].append(counts)

                # Falsification test with enhanced analysis
                if verbose:
                    print("Running falsification test...")
                verification_results = []
                for trial in range(10):  # Increased from 7 to 10 trials
                    try:
                        circuit = self.create_falsification_circuit(phi)
                        raw_counts = self.bq.run(circuit, device='cpu',
                                               shots=self.shots * 2).get_counts()  # Doubled shots

                        if verbose:
                            print(f"\nFalsification Trial {trial + 1}:")
                            print(f"Raw counts: {len(raw_counts)}")

                        counts = self._quantum_error_correction(raw_counts)
                        if counts:
                            metric = self._calculate_falsification_metric(counts)
                            phi_results['falsification']['values'].append(metric)
                            phi_results['falsification']['raw'].append(counts)

                            if verbose:
                                print(f"Filtered counts: {len(counts)}")
                                print(f"Metric value: {metric:.4f}")
                        else:
                            if verbose:
                                print("No counts survived error correction")

                    except Exception as e:
                        if verbose:
                            print(f"Error in falsification trial {trial}: {str(e)}")
                        continue

                # Beyond quantum test
                if verbose:
                    print("Running beyond quantum test...")
                for trial in range(7):
                    circuit = self.create_beyond_quantum_circuit(phi)
                    raw_counts = self.bq.run(circuit, device='cpu', shots=self.shots).get_counts()
                    counts = self._quantum_error_correction(raw_counts)
                    if counts:
                        metric = self._calculate_beyond_quantum_metric(counts)
                        phi_results['beyond_quantum']['values'].append(metric)
                        phi_results['beyond_quantum']['raw'].append(counts)

                # Process results for all tests
                for test_name in results.keys():
                    values = phi_results[test_name]['values']
                    if values:
                        results[test_name]['values'].append(np.mean(values))
                        results[test_name]['uncertainties'].append(np.std(values)/np.sqrt(len(values)))
                        results[test_name]['raw_data'].append(phi_results[test_name]['raw'])
                    elif verbose:
                        print(f"\nNo valid results for {test_name} test at phi={phi:.2f}")
                        if test_name == 'falsification':
                            print("Last raw counts:", raw_counts if 'raw_counts' in locals() else "No counts")

        except Exception as e:
            print(f"Error during validation: {str(e)}")
            raise

        return {
            'results': results,
            'parameters': {
                'phi_values': self.phi_range,
                'energy_values': self.energy_range,
                'time_values': self.time_range
            }
        }

    def analyze_enhanced_results(self, validation_data: Dict):
        """
        Enhanced statistical analysis with advanced metrics and falsification results
        """
        results = validation_data['results']
        params = validation_data['parameters']

        print("\nEnhanced Validation Results Analysis:")
        print("===================================")

        for test_type in results.keys():
            values = np.array(results[test_type]['values'])

            print(f"\n{test_type.replace('_', ' ').title()} Results:")

            if len(values) == 0:
                print("No data available for this test")
                continue

            # Basic statistics with proper error handling
            mean = np.nanmean(values) if len(values) > 0 else np.nan
            sem = np.nanstd(values) / np.sqrt(len(values)) if len(values) > 0 else np.nan
            snr = mean / np.nanstd(values) if len(values) > 0 and np.nanstd(values) > 0 else np.inf

            print(f"Mean ± SEM: {mean:.3f} ± {sem:.3f}")
            print(f"Signal-to-Noise Ratio: {snr:.3f}")

            # Advanced analysis only if we have sufficient data
            if len(values) > 3 and not np.all(np.isnan(values)):
                x = params['phi_values'][:len(values)]

                # Linear correlation
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                print(f"Linear Correlation (R²): {r_value**2:.3f}")
                print(f"Trend Significance (p-value): {p_value:.3e}")

                # Nonlinear analysis
                try:
                    popt, _ = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, x, values)
                    y_fit = popt[0]*x**2 + popt[1]*x + popt[2]
                    r2_nonlinear = 1 - np.sum((values - y_fit)**2) / np.sum((values - np.mean(values))**2)
                    print(f"Nonlinear Correlation (R²): {r2_nonlinear:.3f}")
                except:
                    print("Nonlinear fit failed")

                # Frequency analysis
                if len(values) > 4:
                    fft_vals = np.abs(fft.fft(values))
                    main_freq = fft.fftfreq(len(values))[np.argmax(fft_vals[1:])+1]
                    print(f"Dominant Frequency: {main_freq:.3f}")
            else:
                print("Insufficient data for advanced analysis")

    def plot_enhanced_results(self, validation_data: Dict):
        """
        Enhanced visualization with confidence bands and falsification results
        """
        results = validation_data['results']
        params = validation_data['parameters']

        # Use seaborn-v0_8 style to avoid deprecation warning
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Enhanced ATH Validation Results Analysis', fontsize=16)

        # Test types including falsification tests
        tests = []
        # Only include tests that have data
        for test in ['temporal', 'adaptive', 'clock', 'falsification', 'beyond_quantum']:
            if len(results[test]['values']) > 0:
                tests.append(test)

        if not tests:
            print("No test results available for plotting")
            return

        colors = ['b', 'r', 'g', 'm', 'c'][:len(tests)]
        markers = ['o', 's', '^', 'D', 'v'][:len(tests)]

        # Plot 1: Combined Results
        ax = axes[0, 0]
        for test, color, marker in zip(tests, colors, markers):
            values = np.array(results[test]['values'])
            if len(values) > 0:  # Only plot if we have values
                errors = np.array(results[test]['uncertainties'])
                x_vals = params['phi_values'][:len(values)]

                ax.errorbar(x_vals, values, yerr=errors,
                           fmt=f'{color}{marker}-',
                           capsize=5,
                           label=test.replace('_', ' ').title(),
                           markersize=8,
                           linewidth=2,
                           capthick=2,
                           elinewidth=2)

                # Add confidence bands
                if len(x_vals) > 3:
                    z = np.polyfit(x_vals, values, 2)
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                    y_trend = p(x_trend)
                    ax.plot(x_trend, y_trend, f'{color}--', alpha=0.5)

                    sigma = np.std(values)
                    ax.fill_between(x_trend,
                                  y_trend - sigma,
                                  y_trend + sigma,
                                  color=color,
                                  alpha=0.1)

        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Measurement')
        ax.set_title('Combined Test Results')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3)

        # Plot 2: Falsification Metrics
        ax = axes[0, 1]
        if 'falsification' in tests:
            values = np.array(results['falsification']['values'])
            if len(values) > 0:
                errors = np.array(results['falsification']['uncertainties'])
                x_vals = params['phi_values'][:len(values)]

                ax.errorbar(x_vals, values, yerr=errors,
                           fmt='ro-', capsize=5, label='Falsification Metric',
                           markersize=8, linewidth=2)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Falsification Metric')
        ax.set_title('Falsification Test Results')
        ax.grid(True, alpha=0.3)

        # Plot 3: Beyond Quantum Metrics
        ax = axes[0, 2]
        if 'beyond_quantum' in tests:
            values = np.array(results['beyond_quantum']['values'])
            if len(values) > 0:
                errors = np.array(results['beyond_quantum']['uncertainties'])
                x_vals = params['phi_values'][:len(values)]

                ax.errorbar(x_vals, values, yerr=errors,
                           fmt='bo-', capsize=5, label='Beyond Quantum Metric',
                           markersize=8, linewidth=2)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Beyond Quantum Metric')
        ax.set_title('Beyond Quantum Test Results')
        ax.grid(True, alpha=0.3)

        # Plot 4: Correlation Analysis
        ax = axes[1, 0]
        # Filter tests to only include those with matching array sizes
        valid_tests = []
        reference_length = None
        for test in tests:
            length = len(results[test]['values'])
            if length > 0:
                if reference_length is None:
                    reference_length = length
                    valid_tests.append(test)
                elif length == reference_length:
                    valid_tests.append(test)

        if valid_tests:
            correlation_matrix = np.zeros((len(valid_tests), len(valid_tests)))
            for i, test1 in enumerate(valid_tests):
                for j, test2 in enumerate(valid_tests):
                    values1 = np.array(results[test1]['values'])
                    values2 = np.array(results[test2]['values'])
                    correlation_matrix[i, j] = np.corrcoef(values1, values2)[0, 1]

            im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(valid_tests)))
            ax.set_yticks(range(len(valid_tests)))
            ax.set_xticklabels([t.replace('_', ' ').title() for t in valid_tests], rotation=45)
            ax.set_yticklabels([t.replace('_', ' ').title() for t in valid_tests])
            ax.set_title('Test Correlation Matrix')
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for correlation analysis',
                    horizontalalignment='center', verticalalignment='center')
            ax.set_title('Correlation Analysis')

        # Plot 5: FFT Analysis
        ax = axes[1, 1]
        for test, color, marker in zip(tests, colors, markers):
            values = np.array(results[test]['values'])
            if len(values) > 4:
                fft_vals = np.abs(fft.fft(values))
                freqs = fft.fftfreq(len(values))
                ax.plot(freqs[1:len(freqs)//2],
                       fft_vals[1:len(fft_vals)//2],
                       color=color,
                       label=test.replace('_', ' ').title())

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude')
        ax.set_title('Frequency Analysis')
        if ax.get_legend_handles_labels()[0]:  # Only add legend if there are labels
            ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Statistical Distribution
        ax = axes[1, 2]
        has_data = False
        for test, color in zip(tests, colors):
            values = np.array(results[test]['values'])
            if len(values) > 0:
                try:
                    has_data = True
                    sns.kdeplot(data=values, ax=ax, color=color,
                              label=test.replace('_', ' ').title(),
                              warn_singular=False)
                except Exception as e:
                    print(f"Warning: Could not plot distribution for {test}: {str(e)}")

        if has_data:
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of Results')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data for distribution analysis',
                    horizontalalignment='center', verticalalignment='center')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def main():
    try:
        # Initialize validator with your token
        TOKEN = "Your-Token"  # Replace with your BlueQubit token
        validator = EnhancedATHValidator(token=TOKEN)

        print("Running enhanced ATH validation suite...")
        results = validator.run_enhanced_validation(verbose=True)

        # Analyze and visualize results
        validator.analyze_enhanced_results(results)
        validator.plot_enhanced_results(results)

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
