#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots amplitude spctrum of the input trace twice-integrated 
reads input npz file
Usage: python plot_twice_integrated_spectrum.py stack_A.npz

Created on Wed Feb 11 17:10:17 2026

@author: stefan
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


path = sys.argv[1] if len(sys.argv) > 1 else "stack_A.npz"

npz = np.load(path)              # do NOT allow_pickle; we only read numeric arrays
dt = float(npz["dt"])
x  = np.asarray(npz["stack"], dtype=float)

# (optional but usually helpful) remove DC offset
x = x - x.mean()

# FFT (one-sided)
X = np.fft.rfft(x)
f = np.fft.rfftfreq(x.size, d=dt)          # Hz
w = 2 * np.pi * f                          # rad/s

# Twice time integration in frequency domain: divide by (i*w)^2 = -w^2
# For amplitude spectrum, just divide by w^2 (skip DC to avoid /0)
amp = np.abs(X).copy()
amp[1:] = amp[1:] / (w[1:] ** 2)
amp[0] = np.nan  # undefined at DC for integration

plt.figure()
plt.loglog(f[1:], amp[1:])
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(ω)| / ω² (arb. units)")
plt.title("Amplitude spectrum of twice time-integrated signal")
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()
