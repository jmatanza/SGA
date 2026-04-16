import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

import sys
from pathlib import Path

# localizar engine
cwd = Path.cwd().resolve()
candidate_roots = [cwd, *cwd.parents[:5]]

project_root = next(
    (p for p in candidate_roots if (p / "Engine" / "quant_engine").exists()),
    None
)

if project_root is None:
    raise RuntimeError("No se encontró Engine/quant_engine")

engine_root = project_root / "Engine"

if str(engine_root) not in sys.path:
    sys.path.insert(0, str(engine_root))


from quant_engine import run_quantization, snr_vs_bits


T0 = 0.02

B_MIN = 1
B_MAX = 12

FS_MIN = 500
FS_MAX = 8000


def plot_quantization(B_bits, fs):

    t, x, xq, e, levels, Delta, mse = run_quantization(B_bits, fs, T0)
    bits, snr_exp, snr_theory = snr_vs_bits(fs, T0, b_min=1, b_max=12)

    fig, axs = plt.subplots(4, 1, figsize=(11, 12), constrained_layout=True)

    # ============================================================
    # 1) Señal original + niveles de cuantificación
    # ============================================================
    axs[0].plot(t, x, linewidth=1.5)
    for lev in levels:
        axs[0].axhline(lev, linestyle=':', linewidth=1.0)
    axs[0].grid(True)
    axs[0].set_xlabel("Tiempo (s)")
    axs[0].set_ylabel("Amplitud")
    axs[0].set_title(
        f"Señal original y niveles de cuantificación "
        f"(B={B_bits} bits, L={2**B_bits}, Δ={Delta:.3f}, fs={fs} Hz)"
    )
    axs[0].set_xlim(t[0], t[-1])

    # ============================================================
    # 2) Señal cuantificada
    # ============================================================
    axs[1].plot(t, x, linewidth=1.0, label="Original")
    axs[1].step(t, xq, where='post', linewidth=2.0, label="Cuantificada")
    step_idx = slice(0, None, max(1, len(t)//50))
    axs[1].stem(t[step_idx], xq[step_idx], basefmt=" ", label="Muestras cuantificadas")
    axs[1].grid(True)
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("Amplitud")
    axs[1].set_title("Señal cuantificada (staircase) vs original")
    axs[1].set_xlim(t[0], t[-1])
    axs[1].legend()

    # ============================================================
    # 3) Error de cuantificación
    # ============================================================
    axs[2].plot(t, e, linewidth=1.5)
    axs[2].axhline( Delta/2, linestyle="--", linewidth=1.2, label="±Δ/2 (teórico)")
    axs[2].axhline(-Delta/2, linestyle="--", linewidth=1.2)
    axs[2].grid(True)
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].set_ylabel("Error")
    axs[2].set_title(f"Error de cuantificación: e(t)=x_q(t)-x(t)\nMSE = {mse:.6e}")
    axs[2].set_xlim(t[0], t[-1])
    axs[2].legend()

    # ============================================================
    # 4) SNR de cuantificación vs número de bits
    # ============================================================
    axs[3].plot(bits, snr_exp, 'o-', linewidth=1.8, label="SNR experimental")
    axs[3].plot(bits, snr_theory, '--', linewidth=1.5, label=r"SNR teórica $\approx 6.02B + 1.76$")
    axs[3].axvline(B_bits, color='k', linestyle=':', linewidth=1.2, label=f"B seleccionado = {B_bits}")
    axs[3].grid(True)
    axs[3].set_xlabel("Número de bits B")
    axs[3].set_ylabel("SNR (dB)")
    axs[3].set_title("SNR de cuantificación frente al número de bits")
    axs[3].legend()

    plt.show()



def show_gui():

    b_text = widgets.IntText(value=2,description="B")

    fs_slider = widgets.FloatSlider(
        value=2000,
        min=FS_MIN,
        max=FS_MAX,
        step=100,
        description="fs"
    )

    out = widgets.Output()

    def refresh(*_):

        with out:

            clear_output(wait=True)

            plot_quantization(b_text.value, fs_slider.value)

    b_text.observe(refresh,names="value")
    fs_slider.observe(refresh,names="value")

    display(b_text,fs_slider,out)

    refresh()