Session log — RomaApr2026 (26 Apr 2026)

Summary:
- Explored and ran plotting tools for cal_no_epoxy/flat data (ATF files).
- Created several helper scripts and updated plotting code to trim, baseline-correct, normalize, and display scaling factors.

Actions taken:
1. Opened and inspected `Codes/plot_together.py`.
2. Located flist: `cal_no_epoxy/flat/flist.txt`.
3. Created `cal_no_epoxy/flat/plot_save.py` (non-interactive save) and ran it; saved `plot_together_saved.png` and later `all_ch03.pdf`.
4. Created `cal_no_epoxy/flat/plot_two.py` to plot `stacked_ch2.atf` and `stacked_ch3.atf` interactively.
5. Created `cal_no_epoxy/flat/plot_two_norm.py` to trim (0-0.000125 s), baseline-correct (5-25 us), normalize, and plot twin y-axes; ran it interactively.
6. Copied `plot_two_norm.py` into `Codes/plot_two_norm.py` for reuse.
7. Modified `Codes/plot_two_norm.py` to change baseline window (5-25 µs) and add a textbox with inverse normalization factors and ratio `C`.
8. Created `cal_no_epoxy/flat/plot_trim_ch2.py` and `plot_trim_ch3.py` and used them to display trimmed views of `stacked_ch2.atf` and `stacked_ch3.atf` (0–0.000125 s).
9. Updated `Codes/plot_two_norm.py` to add amplitude multipliers `amp_02=31.6` and `amp_03=2.0`, relabel textbox entries to `Ch02 (Acc)` and `Ch03 (PZ)`, and recompute `C = (Ch03*amp_03)/(Ch02*amp_02)`; re-ran interactively.

Files created/edited (workspace-relative):
- cal_no_epoxy/flat/plot_save.py (created)
- cal_no_epoxy/flat/plot_two.py (created)
- cal_no_epoxy/flat/plot_two_norm.py (created)
- cal_no_epoxy/flat/plot_trim_ch2.py (created)
- cal_no_epoxy/flat/plot_trim_ch3.py (created)
- cal_no_epoxy/flat/plot_together_saved.png (output)
- cal_no_epoxy/flat/all_ch03.pdf (output)
- Codes/plot_two_norm.py (created/edited)
- Codes/plot_two.py (copied/created)

Representative commands run:
- cd cal_no_epoxy/flat && python3 /home/stefan/Desktop/Calibration/RomaApr2026/Codes/plot_together.py
- cd cal_no_epoxy/flat && python3 plot_save.py all_ch03.pdf
- cd cal_no_epoxy/flat && python3 plot_two.py
- cd cal_no_epoxy/flat && python3 plot_two_norm.py
- cd cal_no_epoxy/flat && python3 plot_trim_ch2.py
- cd cal_no_epoxy/flat && python3 plot_trim_ch3.py

Outputs to check (in folder):
- cal_no_epoxy/flat/all_ch03.pdf
- cal_no_epoxy/flat/plot_together_saved.png

Notes / next steps:
- If you want full reproducible record, I can: (a) commit the created/edited scripts to git, (b) create a tarball of the `cal_no_epoxy/flat` folder with outputs, or (c) save an expanded session transcript (terminal history + timestamps).
- I can also add a simple `run_all.sh` to reproduce the plots and outputs automatically.

End of log.
