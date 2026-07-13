# Codes Overview

## Main scripts in folder *Code*

| File | Description |
| --- | --- |
| `flatten_bsf.py` | Copies or moves nested BSF files into one folder with timestamped filenames. |
|  usage:          | python flatten_bsf.py /data/2026 /data/flat --prefix survey
|                  | python flatten_bsf.py /data/2026/02/17/14/42 /data/flat --prefix survey
| `richter_bsf_to_atf.py` | Converts Richter/ASC BSF waveform files into InSite-compatible ATF files. |
| `stack_atf.py` | Averages ATF traces listed in a text file and writes `stack.atf`. |
| `plot_two.py` | Compares two ATF traces with baseline correction, optional shift/inversion, scaling, and normalization. |
|   usage:      | python plot_two.py @plot_two_input.txt
| `spectral_calibrate.py` | Estimates frequency-dependent calibration between paired ATF channels and writes CSV/PNG outputs. |
| `filt_atf.py` | Applies FFT-based high-pass, low-pass, or band-pass filtering to one or more ATF-like traces. |

## Input file examples
plot_two_input_july.txt

## Accessory scripts

| File | Description |
| --- | --- |
| `bandpass_clip_tukey_atf.py` | Bandpass filters an ATF trace, clips a selected time window, applies a Tukey taper, and writes a new ATF. |
| `chirp_log.py` | Generates a logarithmic up-down chirp waveform for function-generator calibration tests. |
| `classify_atf.py` | Interactive viewer to classify ATF traces as good or bad. |
| `deconvolve_atf_with_proxy2.py` | Deconvolves an ATF trace using a proxy impulse response stored in NPZ. |
| `plot_any_flexible.py` | Flexible plotting utility for ATF, NPY, and NPZ trace files. |
| `plot_atf_list.py` | Interactive sequential viewer for ATF files listed in a text file. |
| `plot_cool.py` | Plots a saved chirp waveform, its spectrum, and instantaneous frequency. |
| `plot_flist2.py` | Plots ATF traces from `flist.txt` in a grid of subplots. |
| `plot_seq.py` | Interactively adds aligned NPZ traces to a plot, optionally showing a stack first. |
| `plot_together.py` | Overlays all traces listed in `flist.txt` on one common plot. |
| `prep.py` | Interactive two-trace ATF preprocessing: clip, baseline subtract, optional demean/detrend, taper, and save TXT. |
| `read_atf_function.py` | Small reusable function for reading ATF data and sampling interval. |
| `spectrum.py` | Plots the amplitude spectrum of a twice-integrated stacked NPZ trace. |
| `stack_peak_window_localxcorr.py` | Aligns selected ATF events by peak window and optional local cross-correlation, then stacks them. |
| `tukey.py` | Interactively crops and Tukey-tapers an ATF or NPZ trace, then saves the result. |
| `tukey_orig.py` | Older NPZ-only interactive crop and Tukey taper script. |
