import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ipywidgets import Button, FloatSlider, HBox, Output
from IPython.display import display

cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents[:5]]

project_root = next(
    (p for p in candidate_roots if (p / "Engine" / "nyquist_engine").exists()),
    None,
)
if project_root is None:
    raise RuntimeError(f"No se encontró 'Engine/nyquist_engine' desde: {cwd}")

engine_root = project_root / "Engine"
if str(engine_root) not in sys.path:
    sys.path.insert(0, str(engine_root))

from nyquist_engine import compute_nyquist_simulation


FS = 300.0
F1_MIN = 10.0
F1_MAX = 400.0
F1_STEP = 5.0

out = Output()


def update_plot(f1_val):
    with out:
        out.clear_output(wait=True)

        data = compute_nyquist_simulation(
            f1_val=f1_val,
            fs=FS,
            T0=0.1,
            f_cont=10_000.0,
            K=3,
            Nfft=8192,
        )

        fs = data["fs"]
        K = data["K"]
        t_cont = data["t_cont"]
        n = data["n"]
        x_cont = data["x_cont"]
        x_samp = data["x_samp"]
        X_cont = data["X_cont"]
        f_axis_cont = data["f_axis_cont"]
        X_samp = data["X_samp"]
        f_axis_samp = data["f_axis_samp"]
        f_plot = data["f_plot"]
        Xk_list = data["Xk_list"]
        ymax = data["ymax"]
        x_rec = data["x_rec"]
        aliasing = data["aliasing"]

        alias_text = "ALIASING" if aliasing else "SIN ALIASING"
        alias_color = "C3" if aliasing else "C2"

        fig, axs = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)

        # (1) Tiempo
        axs[0].plot(t_cont, x_cont, linewidth=1.5)
        axs[0].stem(n, x_samp, linefmt='C3-', markerfmt='C3o', basefmt=' ')
        axs[0].set_title(f"Muestreo en tiempo (f1 = {f1_val:g} Hz, fs = {fs:g} Hz)")
        axs[0].set_xlabel("Tiempo (s)")
        axs[0].set_ylabel("Amplitud")
        axs[0].grid(True)

        # (2) Espectros
        axs[1].plot(f_axis_cont, X_cont, linewidth=1.2, label="Original (rejilla fina)")
        axs[1].plot(f_axis_samp, X_samp, linewidth=1.5, linestyle='--', label="Muestreada (discreta)")
        axs[1].set_xlim(-500, 500)
        axs[1].set_title("Espectros (FFT)")
        axs[1].set_xlabel("Frecuencia (Hz)")
        axs[1].set_ylabel("Magnitud")
        axs[1].grid(True)
        axs[1].legend()

        # (3) Réplicas
        ax = axs[2]
        ax.axvspan(-fs / 2, fs / 2, alpha=0.25)
        for k, Xk in Xk_list:
            ax.plot(f_plot, Xk, linewidth=2, label=f"k={k}")
        ax.axvline(fs / 2, linestyle='--', linewidth=1.5)
        ax.axvline(-fs / 2, linestyle='--', linewidth=1.5)
        ax.set_xlim(-K * fs, K * fs)
        ax.set_ylim(0, ymax)

        tick_positions = list(range(-K, K + 1))
        tick_positions = [k * fs for k in tick_positions]
        tick_labels = []
        for k in range(-K, K + 1):
            if k == 0:
                tick_labels.append("0")
            elif k == 1:
                tick_labels.append("Fs")
            elif k == -1:
                tick_labels.append("-Fs")
            else:
                tick_labels.append(f"{k}Fs")

        secax = ax.secondary_xaxis('top')
        secax.set_xticks(tick_positions)
        secax.set_xticklabels(tick_labels)
        secax.set_xlabel("Múltiplos de Fs")

        ax.set_title("Réplicas desde x(t) + 1ª zona de Nyquist")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("|X_s(f)|")
        ax.grid(True)
        ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        ax.text(
            0.02, 0.92, alias_text,
            transform=ax.transAxes,
            color=alias_color,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
        )

        # (4) Reconstrucción
        axs[3].plot(t_cont, x_cont, linewidth=1.2, label="x(t) original")
        axs[3].plot(t_cont, x_rec, linewidth=1.8, label=r"$\hat{x}(t)$ reconstruida (LPF $f_c=f_s/2$)")
        axs[3].set_title("Reconstrucción ideal filtrando la 1ª zona de Nyquist")
        axs[3].set_xlabel("Tiempo (s)")
        axs[3].set_ylabel("Amplitud")
        axs[3].grid(True)
        axs[3].legend()

        plt.show()


f1_slider = FloatSlider(value=50.0, min=F1_MIN, max=F1_MAX, step=F1_STEP, description='f1 (Hz):')
btn_minus = Button(description='-')
btn_plus = Button(description='+')

def on_minus(_):
    f1_slider.value = max(f1_slider.min, f1_slider.value - f1_slider.step)

def on_plus(_):
    f1_slider.value = min(f1_slider.max, f1_slider.value + f1_slider.step)

btn_minus.on_click(on_minus)
btn_plus.on_click(on_plus)

f1_slider.observe(lambda c: update_plot(c['new']), names='value')

controls = HBox([btn_minus, f1_slider, btn_plus])

def show_gui():
    display(controls, out)
    update_plot(f1_slider.value)