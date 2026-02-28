#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strike/Dip/Rake -> DC moment tensor -> P/T/B axes -> assumed stress tensor
-> Beachball with projected P & T axes
-> Mohr diagram (POSITIVE HALF ONLY) with all three circles + Coulomb line
   AND the fault-plane stress point (σn, τ) for the given focal mechanism.

Coordinates: NED (North, East, Down)
Convention: compression positive (geomechanics)
Assumption: principal stress axes align with P/B/T from the mechanism:
  σ1 || P,  σ2 || B,  σ3 || T
"""

import math
import numpy as np

from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Optional beachball (ObsPy)
_HAS_OBSPY = False
try:
    from obspy.imaging.beachball import beach as obspy_beach
    _HAS_OBSPY = True
except Exception:
    _HAS_OBSPY = False


# --------------------------- helpers ---------------------------

def deg2rad(x): return x * math.pi / 180.0
def rad2deg(x): return x * 180.0 / math.pi

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v.copy() if n == 0 else (v / n)

def az_plunge_from_ned(v: np.ndarray):
    """
    Vector in NED -> (azimuth_deg, plunge_deg)
    Azimuth clockwise from North [0..360)
    Plunge positive downward
    """
    v = unit(v)
    n, e, d = float(v[0]), float(v[1]), float(v[2])
    az = math.atan2(e, n)
    az_deg = (rad2deg(az) + 360.0) % 360.0
    h = math.hypot(n, e)
    pl = math.atan2(d, h)
    pl_deg = rad2deg(pl)
    return az_deg, pl_deg

def decompose_symmetric_tensor(S: np.ndarray):
    """Return eigenvectors V (cols) and diagonal D sorted descending."""
    w, v = np.linalg.eigh(S)          # ascending
    idx = np.argsort(w)[::-1]         # descending
    w = w[idx]
    v = v[:, idx]
    return v, np.diag(w)


# --------------------------- SDR -> moment tensor -> P/T/B ---------------------------

def sdr_to_moment_tensor_ned(strike, dip, rake, M0=1.0):
    """
    Convert strike/dip/rake (degrees) to a normalized double-couple moment tensor in NED.
    (Sign conventions vary across toolboxes; P/T/B directions remain consistent for typical use.)
    """
    s = deg2rad(strike)
    d = deg2rad(dip)
    r = deg2rad(rake)

    sd = math.sin(d); cd = math.cos(d)
    s2d = math.sin(2*d); c2d = math.cos(2*d)
    ss = math.sin(s); cs = math.cos(s)
    s2s = math.sin(2*s); c2s = math.cos(2*s)
    sr = math.sin(r); cr = math.cos(r)

    # Components in NED: Mnn, Mee, Mdd, Mne, Mnd, Med
    Mnn = - (sd*cr*s2s + s2d*sr*cs*cs)
    Mee =   (sd*cr*s2s - s2d*sr*ss*ss)
    Mdd =   (s2d*sr)
    Mne =   (sd*cr*c2s + 0.5*s2d*sr*s2s)
    Mnd = - (cd*cr*cs + c2d*sr*ss)
    Med = - (cd*cr*ss - c2d*sr*cs)

    M = np.array([
        [Mnn, Mne, Mnd],
        [Mne, Mee, Med],
        [Mnd, Med, Mdd]
    ], dtype=float)

    fn = np.linalg.norm(M)
    if fn > 0:
        M = (M / fn) * float(M0)
    return M

def ptb_from_dc_moment(M: np.ndarray):
    """
    For DC moment tensor, with eigenvalues sorted descending:
    columns are [T, B, P].
    """
    V, D = decompose_symmetric_tensor(M)
    T = V[:, 0]
    B = V[:, 1]
    P = V[:, 2]
    return P, T, B, V, D


# --------------------------- stress tensor construction ---------------------------

def sigma2_from_shape_ratio(sigma1, sigma3, R):
    """R=(σ2-σ3)/(σ1-σ3) -> σ2=σ3+R(σ1-σ3)"""
    return float(sigma3) + float(R) * (float(sigma1) - float(sigma3))

def build_stress_tensor_from_axes(P, B, T, sigma1, sigma2, sigma3):
    """Assume σ1||P, σ2||B, σ3||T."""
    V = np.column_stack([unit(P), unit(B), unit(T)])
    D = np.diag([float(sigma1), float(sigma2), float(sigma3)])
    S = V @ D @ V.T
    return 0.5 * (S + S.T), V, D


# --------------------------- fault-plane traction (σn, τ) ---------------------------

def fault_normal_and_slip_ned(strike_deg, dip_deg, rake_deg):
    """
    Build fault plane unit normal n and slip vector s in NED for the given strike/dip/rake.
    Strike measured clockwise from North. Dip is down to the right of strike.
    Rake measured within the plane from strike direction toward dip direction.
    """
    phi = deg2rad(strike_deg)
    dip = deg2rad(dip_deg)
    rake = deg2rad(rake_deg)

    # Plane normal (NED) for strike φ, dip δ
    # n = [-sinδ*sinφ, sinδ*cosφ, cosδ]
    n = np.array([
        -math.sin(dip) * math.sin(phi),
         math.sin(dip) * math.cos(phi),
         math.cos(dip)
    ], dtype=float)
    n = unit(n)

    # Strike unit vector (horizontal)
    sv = np.array([math.cos(phi), math.sin(phi), 0.0], dtype=float)
    sv = unit(sv)

    # Dip-direction unit vector (pointing down-dip within the plane)
    # horizontal dip azimuth = φ+90°
    hd = np.array([-math.sin(phi), math.cos(phi), 0.0], dtype=float)  # unit
    dv = np.array([hd[0] * math.cos(dip), hd[1] * math.cos(dip), math.sin(dip)], dtype=float)
    dv = unit(dv)

    # Slip vector in plane
    slip = sv * math.cos(rake) + dv * math.sin(rake)
    slip = unit(slip)

    # Ensure slip is within plane numerically
    slip = unit(slip - np.dot(slip, n) * n)

    return n, slip

def traction_normal_shear(S: np.ndarray, n: np.ndarray):
    """
    Given stress tensor S and plane normal n (unit),
    traction t = S n, normal stress σn = n·t, shear magnitude τ = ||t - σn n||.
    """
    n = unit(n)
    t = S @ n
    sig_n = float(np.dot(n, t))
    shear_vec = t - sig_n * n
    tau = float(np.linalg.norm(shear_vec))
    return sig_n, tau, t, shear_vec


# --------------------------- beachball projection (for P/T markers) ---------------------------

def lower_hemisphere_equal_area_xy(az_deg, pl_deg, R=100.0):
    """
    Lower-hemisphere equal-area (Schmidt) projection to x,y.
    az: clockwise from North; pl: +down.
    Returns x,y scaled to radius R.
    """
    # colatitude from down-going direction
    colat = deg2rad(90.0 - pl_deg)  # 0 at vertical down, 90 at horizontal
    r = math.sqrt(2.0) * math.sin(colat / 2.0)  # 0..1
    az = deg2rad(az_deg)
    x = R * r * math.sin(az)
    y = R * r * math.cos(az)
    return x, y


# --------------------------- plotting ---------------------------

def plot_beachball(ax, strike, dip, rake, P, T, B):
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    # Base circle / beachball
    if _HAS_OBSPY:
        bb = obspy_beach([strike, dip, rake], xy=(0, 0), width=200, linewidth=1.2)
        ax.add_collection(bb)
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        R = 100.0
        ax.set_title("Focal Mechanism", fontsize=8)
    else:
        # simple outline circle if ObsPy missing
        R = 100.0
        th = np.linspace(0, 2 * math.pi, 361)
        ax.plot(R * np.cos(th), R * np.sin(th), linewidth=1.2)
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.set_title("Projected P/T axes" , fontsize=8)

    # Axis annotations (az/plunge)
    paz, ppl = az_plunge_from_ned(P)
    taz, tpl = az_plunge_from_ned(T)
    baz, bpl = az_plunge_from_ned(B)

    # Project P and T onto beachball and plot markers
    px, py = lower_hemisphere_equal_area_xy(paz, ppl, R=R)
    tx, ty = lower_hemisphere_equal_area_xy(taz, tpl, R=R)

    ax.plot(px, py, marker="o", markersize=6)
    ax.text(px, py, "  P", fontsize=6, va="center", ha="left")

    ax.plot(tx, ty, marker="o", markersize=6)
    ax.text(tx, ty, "  T", fontsize=6, va="center", ha="left")

    # Info text
    txt = (
        f"P az={paz:6.1f}°, pl={ppl:5.1f}°\n"
        f"T az={taz:6.1f}°, pl={tpl:5.1f}°\n"
        f"B az={baz:6.1f}°, pl={bpl:5.1f}°"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6, va="bottom", ha="left")

def plot_mohr_all_positive(ax, sigma1, sigma2, sigma3, mu=0.85, focal_point=None):
    """
    Plot POSITIVE HALF ONLY: τ >= 0
    Draw all three Mohr circles upper halves only, plus Coulomb line τ=μσn (positive),
    plus an optional focal-plane point (σn, τ).
    """
    ax.clear()

    # Ensure ordering σ1 >= σ2 >= σ3
    s = np.array([sigma1, sigma2, sigma3], dtype=float)
    s.sort()
    sigma3, sigma2, sigma1 = s[0], s[1], s[2]

    circles = [
        ("σ1–σ3", sigma1, sigma3, 2.2),
        ("σ1–σ2", sigma1, sigma2, 1.6),
        ("σ2–σ3", sigma2, sigma3, 1.6),
    ]

    theta = np.linspace(0.0, 0.5 * math.pi, 900)  # upper half only
    all_sn, all_tau = [], []

    for label, si, sj, lw in circles:
        C = 0.5 * (si + sj)
        R = 0.5 * abs(si - sj)

        sn = C + R * np.cos(2.0 * theta)
        tau = R * np.sin(2.0 * theta)

        ax.plot(sn, tau, linewidth=lw, label=label)
        ax.plot([C], [0.0], marker="o", markersize=5)

        all_sn.append(sn)
        all_tau.append(tau)

    all_sn = np.concatenate(all_sn) if all_sn else np.array([0.0])
    all_tau = np.concatenate(all_tau) if all_tau else np.array([0.0])

    # Mark σ1 σ2 σ3
    ax.plot([sigma1, sigma2, sigma3], [0.0, 0.0, 0.0],
            linestyle="none", marker="o", markersize=7)
    ax.annotate("σ1", (sigma1, 0.0), textcoords="offset points", xytext=(6, 6))
    ax.annotate("σ2", (sigma2, 0.0), textcoords="offset points", xytext=(6, 6))
    ax.annotate("σ3", (sigma3, 0.0), textcoords="offset points", xytext=(6, 6))

    # Coulomb envelope (positive only)
    span = max(all_sn.max(), 1e-9) * 1.15
    x = np.linspace(0.0, span, 200)
    y = mu * x
    ax.plot(x, y, linewidth=1.3)

    # Focal mechanism shear stress point (σn, τ)
    if focal_point is not None:
        sig_n, tau = focal_point
        # Only show if it lies in plotted half-space
        if tau >= 0:
            ax.plot([sig_n], [tau], marker="o", markersize=6)
            ax.annotate(" Fault plane\n (σn, τ)",
                        (sig_n, tau),
                        textcoords="offset points", xytext=(8, 10),
                        ha="left", va="bottom", fontsize=6)

    ax.grid(True, alpha=0.35)
    ax.set_xlabel("Normal stress, σn")
    ax.set_ylabel("Shear stress, τ")
    ax.set_aspect("equal", adjustable="box")

    xmin = min(all_sn.min(), 0.0)
    xmax = max(all_sn.max(), span)
    xr = (xmax - xmin) if (xmax - xmin) > 0 else 1.0
    xmin -= 0.06 * xr
    xmax += 0.06 * xr

    ymax = max(all_tau.max(), y.max(), 1e-9) * 1.15
    # Ensure point visible if it's higher
    if focal_point is not None:
        ymax = max(ymax, focal_point[1] * 1.15)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, ymax)

    ax.set_title(f"Mohr Diagram (μ={mu:.2f})", fontsize=8)
    ax.legend(loc="upper right", frameon=True)


# --------------------------- GUI ---------------------------

class MplPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 4), dpi=120)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Focal Mechanism and Mohr Diagram")
        self.resize(1160, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Controls
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.strike = QtWidgets.QDoubleSpinBox(); self.strike.setRange(0, 360); self.strike.setDecimals(1); self.strike.setValue(315.0)
        self.dip    = QtWidgets.QDoubleSpinBox(); self.dip.setRange(0, 90);  self.dip.setDecimals(1);    self.dip.setValue(60.0)
        self.rake   = QtWidgets.QDoubleSpinBox(); self.rake.setRange(-180, 180); self.rake.setDecimals(1); self.rake.setValue(-90.0)

        self.mu     = QtWidgets.QDoubleSpinBox(); self.mu.setRange(0.0, 2.0); self.mu.setDecimals(3); self.mu.setValue(0.6)

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems([
            "Use shape ratio R (σ2 = σ3 + R(σ1-σ3))",
            "Use explicit σ1, σ2, σ3"
        ])

        self.s1 = QtWidgets.QDoubleSpinBox(); self.s1.setRange(-1e9, 1e9); self.s1.setDecimals(6); self.s1.setValue(1.0)
        self.s3 = QtWidgets.QDoubleSpinBox(); self.s3.setRange(-1e9, 1e9); self.s3.setDecimals(6); self.s3.setValue(0.0)
        self.R  = QtWidgets.QDoubleSpinBox(); self.R.setRange(0.0, 1.0);   self.R.setDecimals(6);  self.R.setValue(0.5)
        self.s2 = QtWidgets.QDoubleSpinBox(); self.s2.setRange(-1e9, 1e9); self.s2.setDecimals(6); self.s2.setValue(0.5)

        self.btn_compute = QtWidgets.QPushButton("Compute + Plot")
        self.btn_export  = QtWidgets.QPushButton("Export Figures (PNG)")

        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMinimumHeight(220)

        form.addRow("Strike (°):", self.strike)
        form.addRow("Dip (°):", self.dip)
        form.addRow("Rake (°):", self.rake)
        form.addRow("Friction μ:", self.mu)
        form.addRow("Stress input mode:", self.mode)
        form.addRow("σ1 (compression +):", self.s1)
        form.addRow("σ3:", self.s3)
        form.addRow("R (0..1):", self.R)
        form.addRow("σ2 (explicit):", self.s2)
        form.addRow(self.btn_compute, self.btn_export)

        left = QtWidgets.QWidget()
        left.setLayout(form)

        # Plots
        self.tabs = QtWidgets.QTabWidget()
        self.pnl_beach = MplPanel()
        self.pnl_mohr  = MplPanel()
        self.tabs.addTab(self.pnl_beach, "Beachball + P/T markers")
        self.tabs.addTab(self.pnl_mohr, "Mohr (τ ≥ 0) + Fault-plane point")

        # Layout
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        left_wrap = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left_wrap)
        left_lay.addWidget(left)
        left_lay.addWidget(QtWidgets.QLabel("Computed results:"))
        left_lay.addWidget(self.out)
        left_lay.setStretch(0, 0)
        left_lay.setStretch(2, 1)

        split.addWidget(left_wrap)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        main_lay = QtWidgets.QVBoxLayout(central)
        main_lay.addWidget(split)

        # Signals
        self.btn_compute.clicked.connect(self.compute_and_plot)
        self.btn_export.clicked.connect(self.export_pngs)
        self.mode.currentIndexChanged.connect(self._update_mode_ui)

        self._update_mode_ui()
        self.compute_and_plot()

    def _update_mode_ui(self):
        use_R = (self.mode.currentIndex() == 0)
        self.R.setEnabled(use_R)
        self.s2.setEnabled(not use_R)

    def compute_and_plot(self):
        strike = float(self.strike.value())
        dip    = float(self.dip.value())
        rake   = float(self.rake.value())
        mu     = float(self.mu.value())

        # 1) SDR -> DC moment tensor
        M = sdr_to_moment_tensor_ned(strike, dip, rake, M0=1.0)

        # 2) P/T/B axes
        P, T, B, Vmt, Dmt = ptb_from_dc_moment(M)

        # Make plunge positive (down) by flipping direction when needed
        def make_plunge_down(v):
            _, pl = az_plunge_from_ned(v)
            return v if pl >= 0 else -v

        P = make_plunge_down(P)
        T = make_plunge_down(T)
        B = make_plunge_down(B)

        # 3) principal stresses
        sigma1 = float(self.s1.value())
        sigma3 = float(self.s3.value())

        if self.mode.currentIndex() == 0:
            R = float(self.R.value())
            sigma2 = sigma2_from_shape_ratio(sigma1, sigma3, R)
        else:
            sigma2 = float(self.s2.value())
            R = (sigma2 - sigma3) / (sigma1 - sigma3) if abs(sigma1 - sigma3) > 1e-12 else float("nan")

        # 4) assumed stress tensor aligned with (P,B,T)
        S, Vs, Ds = build_stress_tensor_from_axes(P, B, T, sigma1, sigma2, sigma3)

        # Fault-plane stress point
        n, slip = fault_normal_and_slip_ned(strike, dip, rake)
        sig_n, tau, tvec, shear_vec = traction_normal_shear(S, n)

        # Report
        paz, ppl = az_plunge_from_ned(P)
        taz, tpl = az_plunge_from_ned(T)
        baz, bpl = az_plunge_from_ned(B)

        Vst, Dst = decompose_symmetric_tensor(S)
        svals = np.diag(Dst)

        msg = []
        msg.append("INPUT focal mechanism (degrees):")
        msg.append(f"  strike={strike:.1f}, dip={dip:.1f}, rake={rake:.1f}")
        msg.append("")
        msg.append("P/T/B axes from DC moment tensor (azimuth°, plunge°; plunge +down):")
        msg.append(f"  P (σ1 dir): az={paz:6.1f}, pl={ppl:6.1f}")
        msg.append(f"  B (σ2 dir): az={baz:6.1f}, pl={bpl:6.1f}")
        msg.append(f"  T (σ3 dir): az={taz:6.1f}, pl={tpl:6.1f}")
        msg.append("")
        msg.append("ASSUMED principal stresses (compression +):")
        msg.append(f"  σ1={sigma1:.6g}, σ2={sigma2:.6g}, σ3={sigma3:.6g}")
        msg.append(f"  Shape ratio R={R:.6f}")
        msg.append("")
        msg.append("Stress tensor S (N,E,D):")
        msg.append(np.array2string(S, formatter={'float_kind': lambda x: f"{x: .6f}"}))
        msg.append("")
        msg.append("Eigenvalues of S (sorted σ1≥σ2≥σ3):")
        msg.append(f"  {svals[0]:.6g}  {svals[1]:.6g}  {svals[2]:.6g}")
        msg.append("")
        msg.append("Fault-plane traction (computed from S and the given strike/dip/rake plane):")
        msg.append(f"  σn = {sig_n:.6g}")
        msg.append(f"  τ  = {tau:.6g}   (plotted on Mohr diagram)")
        msg.append("")
        msg.append(f"Coulomb envelope (positive only): τ = μ σn,  μ={mu:.3f}")
        if not _HAS_OBSPY:
            msg.append("")
            msg.append("NOTE: ObsPy not found; beachball shading is disabled, but P/T markers still plot.")

        self.out.setPlainText("\n".join(msg))

        # Plots
        plot_beachball(self.pnl_beach.ax, strike, dip, rake, P, T, B)
        self.pnl_beach.canvas.draw_idle()

        plot_mohr_all_positive(self.pnl_mohr.ax, sigma1, sigma2, sigma3, mu=mu, focal_point=(sig_n, tau))
        self.pnl_mohr.canvas.draw_idle()

    def export_pngs(self):
        self.pnl_beach.fig.savefig("beachball_pt_projected.png", dpi=300, bbox_inches="tight")
        self.pnl_mohr.fig.savefig("mohr_positive_half_with_fault_point.png", dpi=300, bbox_inches="tight")
        QtWidgets.QMessageBox.information(
            self, "Export complete",
            "Saved:\n  beachball_pt_projected.png\n  mohr_positive_half_with_fault_point.png"
        )


def main():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
