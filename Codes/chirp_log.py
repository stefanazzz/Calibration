import numpy as np
"""
Generate a chirp / sweep (low to high to low frequency)
to upload on the function generator for calibration tests 
S Nielsen March 2026
"""

def tukey_window(N: int, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        n = np.arange(N)
        return 0.5 * (1 - np.cos(2*np.pi*n/(N-1)))
    n = np.arange(N)
    w = np.ones(N)
    edge = int(np.floor(alpha*(N-1)/2))
    if edge > 0:
        k = np.arange(edge)
        w[k] = 0.5 * (1 + np.cos(np.pi*(2*k/(alpha*(N-1)) - 1)))
        w[N-edge:] = w[edge-1::-1]
    return w

def updown_log_chirp(
    N=4096, T=1e-3, f0=10e3, f1=500e3,
    amp=0.95, alpha=0.1, quant16=True
):
    """
    Up-down *log* chirp (exponential sweep):
      first half: f(t) = f0 * (f1/f0)^(t/Th)
      second half: f(t) = f1 * (f0/f1)^((t-Th)/Th)

    log (powerlaw) chirp gives more equal time/energy per octave than a linear chirp.
    """
    if f0 <= 0 or f1 <= 0:
        raise ValueError("f0 and f1 must be > 0 for a log chirp.")
    if f1 <= f0:
        raise ValueError("Require f1 > f0 for an up-down sweep.")

    t = np.arange(N) * (T / N)
    Th = T / 2.0
    dt = T / N

    # instantaneous frequency: log up then log down
    f = np.empty_like(t, dtype=float)
    up = t < Th

    r = f1 / f0
    f[up]  = f0 * (r ** (t[up] / Th))
    f[~up] = f1 * ((1.0 / r) ** ((t[~up] - Th) / Th))

    # integrate frequency to phase
    phase = 2*np.pi * np.cumsum(f) * dt
    x = np.sin(phase)

    # taper to suppress wrap discontinuity
    x *= tukey_window(N, alpha)

    # scale to 16-bit signed
    y = np.round(x * (amp * 32767.0)).astype(np.int64)
    y = np.clip(y, -32768, 32767)

    # optional: align with DF1906 effective 12-bit output (lower 4 bits ignored)
    if quant16:
        y = (np.round(y / 16.0) * 16.0).astype(np.int64)
        y = np.clip(y, -32768, 32767)

    return y.astype(np.int16)

# --- choose T here ---
T = 1e-3  # start with 1 ms (try 0.5 ms if DF1906 accepts it cleanly)
y = updown_log_chirp(N=4096, T=T, f0=10e3, f1=500e3, amp=0.95, alpha=0.1, quant16=True)

np.savetxt("df1906_logchirp_10k_500k_updown_4096.txt", y, fmt="%d")

print("Saved df1906_logchirp_10k_500k_updown_4096.txt")
print(f"On DF1906 set USER waveform frequency to 1/T = {1/T:.6g} Hz (continuous repeat).")
print(f"Effective sample rate Fs = 4096/T = {4096/T:.6g} Sa/s")
