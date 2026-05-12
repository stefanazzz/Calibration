import numpy as np
import matplotlib.pyplot as plt
import math
import os


#def read_atf_new(path: str) -> tuple[np.ndarray, float]:
#    with open(path, "r", errors="ignore") as f:
#        _ = f.readline().strip()
#        header2 = f.readline().strip()
#
#        def get_header_float(name: str) -> float:
#            m = re.search(rf"{re.escape(name)}\s*=\s*([0-9.eE+-]+)", header2)
#            if not m:
#                raise ValueError(f"Missing header field '{name}' in {path}\nHeader: {header2}")
#            return float(m.group(1))
#
#        def get_header_int(name: str) -> int:
#            m = re.search(rf"{re.escape(name)}\s*=\s*(\d+)", header2)
#            if not m:
#                raise ValueError(f"Missing header field '{name}' in {path}\nHeader: {header2}")
#            return int(m.group(1))
#
#        n = get_header_int("TracePoints")
#        tsamp = get_header_float("TSamp")
#        time_units = get_header_float("TimeUnits")
#        amp_to_volts = get_header_float("AmpToVolts")
#        dt = tsamp * time_units
#
#        line = f.readline()
#        while line and "[TraceData]" not in line:
#            line = f.readline()
#
#        if not line:
#            raise ValueError(f"[TraceData] section not found in {path}")
#
#        data = []
#        for _ in range(n):
#            s = f.readline()
#            if not s:
#                break
#            s = s.strip()
#            if s:
#                data.append(float(s))
#
#    x = np.asarray(data, dtype=np.float64) * amp_to_volts
#    return x, dt
##################
def read_atf(fname):
    with open(fname) as f:
        lines = f.readlines()

    # parse metadata line safely
    meta = lines[1].strip().strip(';').split(';')
    fields = {}
    for item in meta:
        if '=' in item:
            k, v = item.split('=', 1)
            fields[k.strip()] = v.strip()

    # handle common key variants
    if 'TSamp' in fields:
        TSamp = float(fields['TSamp'])
    elif 'SamplingInterval' in fields:
        TSamp = float(fields['SamplingInterval'])
    else:
        raise KeyError(f"No TSamp/SamplingInterval found in header of {fname}")

    TimeUnits = float(fields.get('TimeUnits', 1.0))
    dt = TSamp * TimeUnits

    # data start
    i0 = lines.index('[TraceData]\n') + 1
    data = np.loadtxt(lines[i0:])

    t = np.arange(len(data)) * dt
    return t, data, dt


files = [l.strip() for l in open('flist.txt') if l.strip()]
n = len(files)

ncols = 2
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    sharex=True,
    figsize=(10, 1.8 * nrows)
)

# make axes always iterable as 2D array
axes = np.atleast_2d(axes)

for i, f in enumerate(files):
    r = i // ncols
    c = i % ncols
    ax = axes[r, c]

    t, x, dt = read_atf(f)
    ax.plot(t, x, lw=0.8)
    ax.set_title(os.path.basename(f), fontsize=8)
    ax.set_ylabel("AU", fontsize=8)

# hide unused axes (if n is odd)
for i in range(n, nrows * ncols):
    r = i // ncols
    c = i % ncols
    axes[r, c].set_visible(False)

# x-label only on bottom row
for ax in axes[-1, :]:
    ax.set_xlabel('Time (s)')

fig.tight_layout()
plt.show()

