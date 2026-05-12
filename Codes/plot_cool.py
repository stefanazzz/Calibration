import numpy as np
import matplotlib.pyplot as plt

# ---------- USER SETTINGS ----------
filename = "df1906_logchirp_10k_500k_updown_4096.txt"
T = 1e-3  # MUST match generation time
# -----------------------------------

# Load data
y = np.loadtxt(filename).astype(np.int16)
N = len(y)

# Time axis
t = np.arange(N) * (T / N)
Fs = N / T

# --- Plot time waveform ---
plt.figure()
plt.plot(t * 1e3, y)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (int16)")
plt.title("Log Up-Down Chirp (Time Domain)")
plt.grid(True)
plt.show()

# --- Plot amplitude spectrum ---
Y = np.fft.rfft(y)
freq = np.fft.rfftfreq(N, d=1/Fs)

plt.figure()
plt.semilogx(freq, 20*np.log10(np.abs(Y) + 1e-12))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB, arbitrary)")
plt.title("Amplitude Spectrum")
plt.grid(True, which="both")
plt.xlim(1e3, Fs/2)
plt.show()

# --- Estimate instantaneous frequency (numerical derivative of phase) ---
# Reconstruct float signal for phase estimate
x = y.astype(float) / 32767.0
analytic = np.angle(np.fft.ifft(np.fft.fft(x)))  # simple unwrap alternative
phase = np.unwrap(np.angle(np.fft.hilbert(x))) if hasattr(np.fft, "hilbert") else None

# If scipy not available, compute derivative of phase from FFT-based analytic signal:
from scipy.signal import hilbert
analytic = hilbert(x)
phase = np.unwrap(np.angle(analytic))
inst_freq = np.diff(phase) * Fs / (2*np.pi)

plt.figure()
plt.plot(t[:-1] * 1e3, inst_freq)
plt.xlabel("Time (ms)")
plt.ylabel("Instantaneous Frequency (Hz)")
plt.title("Instantaneous Frequency vs Time")
plt.grid(True)
plt.show()

print(f"N = {N}")
print(f"T = {T} s")
print(f"Fs = {Fs/1e6:.3f} MSa/s")
print(f"Nyquist = {Fs/2/1e3:.1f} kHz")
