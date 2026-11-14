import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

def test_device_frequencies(device_id=0, duration=5.0, sample_rate=44100):
    """
    Records audio from a specified device for a given duration and plots
    its frequency spectrum.
    """
    print("=" * 60)
    print(f" FSK Frequency Analyzer - Testing Device {device_id}")
    print("=" * 60)

    # Check device info
    try:
        device_info = sd.query_devices(device_id)
        print(f"Device {device_id} Name: {device_info['name']}")
        print(f"Max Input Channels: {device_info['max_input_channels']}")
    except Exception as e:
        print(f"Warning: Could not query device {device_id} ({e}). Falling back to default input device.")
        device_id = None

    print(f"\nRecording for {duration} seconds...")
    print("TIP: Speak, whistle, or play your 12kHz/13kHz FSK tones now!")
    
    # Record audio (mono)
    num_samples = int(duration * sample_rate)
    try:
        audio = sd.rec(num_samples, samplerate=sample_rate, channels=1, device=device_id, dtype='float32')
        sd.wait()  # Wait until the 5-second recording finishes
    except Exception as e:
        print(f"Error during recording: {e}")
        return

    audio = audio.flatten()
    print("Recording complete. Computing frequency spectrum (FFT)...")

    # Compute Fast Fourier Transform (FFT)
    n = len(audio)
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)
    fft_vals = np.abs(np.fft.rfft(audio))
    
    # Convert magnitude to Decibels (dB) for clear visualization
    # Using maximum to prevent log(0) errors
    fft_db = 20 * np.log10(np.maximum(fft_vals, 1e-6))

    # Plotting the frequency spectrum
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, fft_db, color='#1f77b4', lw=0.8, alpha=0.9)
    
    # Highlight the FSK target frequencies (12 kHz and 13 kHz) as vertical reference lines
    plt.axvline(x=12000, color='r', linestyle='--', alpha=0.7, label='Target 0 (12 kHz)')
    plt.axvline(x=13000, color='g', linestyle='--', alpha=0.7, label='Target 1 (13 kHz)')

    # Formatting the graph
    dev_name = device_info['name'] if 'device_info' in locals() else 'Default Device'
    plt.title(f"Microphone Frequency Response — Device {device_id} ({dev_name})", fontsize=12)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Magnitude (dB)", fontsize=10)
    plt.xlim(0, sample_rate / 2)  # Limit X-axis up to Nyquist frequency (~22.05 kHz)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Save and display graph
    output_filename = "frequency_test_output.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Graph successfully saved to '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    test_device_frequencies(device_id=0, duration=5.0)