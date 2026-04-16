import sys
from pathlib import Path

HELPER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HELPER_DIR.parent

IN_COLAB = False
try:
    from google.colab import output as colab_output
    IN_COLAB = True
except Exception:
    colab_output = None

# Binder/Colab and local: prefer Engine (pre-built .so); fall back to src
_engine_root = PROJECT_ROOT / "Engine"
_src_root = PROJECT_ROOT / "src"
if IN_COLAB or _engine_root.exists():
    PACKAGE_ROOT = _engine_root
else:
    PACKAGE_ROOT = _src_root

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import asyncio
import threading
from io import BytesIO

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from smartgrids_sim import run_link_packet_scenario
from smartgrids_sim.plots import plot_constellation

plt.rcParams['figure.figsize'] = (8, 4)



def compute_tx_rates_bps(bw_hz, m, cp_len, nsub, fec_enabled=False, fec_n=1, fec_k=1):
    bits_per_symbol = float(np.log2(m))
    eta_cp = nsub / (nsub + cp_len) if (nsub + cp_len) > 0 else 0.0
    gross_rate = float(bw_hz) * bits_per_symbol * eta_cp
    eta_fec = (fec_k / fec_n) if fec_enabled and fec_n > 0 else 1.0
    net_rate = gross_rate * eta_fec
    return gross_rate, net_rate

DEFAULTS = {
    'ptx_dbm': 20.0,
    'gtx_dbi': 2.0,
    'grx_dbi': 2.0,
    'packet_length_bytes': 200,
    'path_loss_mode': 'manual',
    'link_loss_db': 110.0,
    'distance_m': 1000.0,
    'freq_mhz': 2400.0,
    'bw_hz': 180e3,
    'nf_db': 5.0,
    'misc_loss_db': 0.0,
    'fading_mode': 'none',
    'Nsub': 64,
    'cp_len': 16,
    'M': 16,
    'n_frames': 20,
    'fec_enabled': False,
    'fec_n': 3,
    'fec_k': 1,
    'n_taps': 4,
    'availability_window': 10,
}


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def run_link_demo(**params):
    return run_link_packet_scenario(
        ptx_dbm=params['ptx_dbm'],
        gtx_dbi=params['gtx_dbi'],
        grx_dbi=params['grx_dbi'],
        packet_length_bytes=params['packet_length_bytes'],
        path_loss_mode=params['path_loss_mode'],
        link_loss_db=params['link_loss_db'],
        distance_m=params['distance_m'],
        freq_hz=params['freq_mhz'] * 1e6,
        bw_hz=params['bw_hz'],
        nf_db=params['nf_db'],
        misc_loss_db=params['misc_loss_db'],
        fading_mode=params['fading_mode'],
        Nsub=params['Nsub'],
        cp_len=params['cp_len'],
        M=params['M'],
        n_frames=params['n_frames'],
        fec_enabled=params['fec_enabled'],
        fec_n=params['fec_n'],
        fec_k=params['fec_k'],
        n_taps=params['n_taps'],
        seed=params.get('seed', 1234),
    )


def render_link_demo(out, iteration=None):
    title = 'Constelacion Tx y Rx superpuestas' if iteration is None else f'Constelacion Tx y Rx | iter={iteration}'
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    plot_constellation(
        out['digital']['tx_symbols'],
        out['digital']['rx_symbols_post_eq'],
        ax=ax,
    )
    ax.set_title(title)
    info = (
        f"<b>Path loss:</b> {out['path_loss_db']:.2f} dB<br>"
        f"<b>SNR nominal:</b> {out['snr_db']:.2f} dB<br>"
        f"<b>SNR instantaneo:</b> {out.get('instant_snr_db', out['snr_db']):.2f} dB<br>"
        f"<b>BER:</b> {out['digital']['ber_with_eq']:.3e}<br>"
        f"<b>Vel. bruta:</b> {compute_tx_rates_bps(out['link_budget']['bw_hz'], out['digital']['config']['M'], out['digital']['config']['cp_len'], out['digital']['config']['Nsub'], False, out['digital']['config']['fec_n'], out['digital']['config']['fec_k'])[0]/1e6:.3f} Mbps<br>"
        f"<b>Vel. util:</b> {compute_tx_rates_bps(out['link_budget']['bw_hz'], out['digital']['config']['M'], out['digital']['config']['cp_len'], out['digital']['config']['Nsub'], out['digital']['config']['fec_enabled'], out['digital']['config']['fec_n'], out['digital']['config']['fec_k'])[1]/1e6:.3f} Mbps<br>"
        f"<b>P(packet OK):</b> {out['packet_success_probability']:.6f}<br>"
        f"<b>Packet received OK:</b> {out['packet_received_ok']}"
    )
    return fig_to_png_bytes(fig), info


def render_history_plots(iter_hist, snr_hist, ber_hist, avail_iter_hist, avail_hist, availability_window):
    fig_snr, ax_snr = plt.subplots(figsize=(4.6, 2.3))
    if iter_hist:
        ax_snr.plot(iter_hist, snr_hist, marker='o')
    ax_snr.set_title('SNR instantaneo ultimas 12 iteraciones')
    ax_snr.set_xlabel('Iteracion')
    ax_snr.set_ylabel('SNR (dB)')
    ax_snr.grid(True, alpha=0.3)

    fig_ber, ax_ber = plt.subplots(figsize=(4.6, 2.3))
    if iter_hist:
        ber_plot = np.asarray(ber_hist, dtype=float)
        ber_plot = np.where(ber_plot > 0.0, ber_plot, np.nan)
        ax_ber.semilogy(iter_hist, ber_plot, marker='o')
    ax_ber.set_title('BER ultimas 12 iteraciones')
    ax_ber.set_xlabel('Iteracion')
    ax_ber.set_ylabel('BER')
    ax_ber.grid(True, which='both', alpha=0.3)

    fig_av, ax_av = plt.subplots(figsize=(4.6, 2.3))
    if avail_iter_hist:
        ax_av.plot(avail_iter_hist, avail_hist, marker='o')
    ax_av.set_title(f'Disponibilidad por bloques de {availability_window} iteraciones')
    ax_av.set_xlabel('Iteracion')
    ax_av.set_ylabel('Disponibilidad (%)')
    ax_av.set_ylim(0.0, 100.0)
    ax_av.grid(True, alpha=0.3)

    return fig_to_png_bytes(fig_snr), fig_to_png_bytes(fig_ber), fig_to_png_bytes(fig_av)


_existing_gui_box = globals().get('GUI_BOX', None)
if _existing_gui_box is not None:
    try:
        _existing_gui_box.close()
    except Exception:
        pass

w_ptx = widgets.FloatText(value=DEFAULTS['ptx_dbm'], description='Ptx (dBm)')
w_gtx = widgets.FloatText(value=DEFAULTS['gtx_dbi'], description='Gtx (dBi)')
w_grx = widgets.FloatText(value=DEFAULTS['grx_dbi'], description='Grx (dBi)')
w_pkt = widgets.IntText(value=DEFAULTS['packet_length_bytes'], description='Pkt (B)')

w_pl_mode = widgets.Dropdown(options=['manual', 'fspl'], value=DEFAULTS['path_loss_mode'], description='Loss mode')
w_loss = widgets.FloatText(value=DEFAULTS['link_loss_db'], description='Loss (dB)')
w_dist = widgets.FloatText(value=DEFAULTS['distance_m'], description='Dist (m)')
w_freq = widgets.FloatText(value=DEFAULTS['freq_mhz'], description='Freq (MHz)')
w_bw = widgets.FloatText(value=DEFAULTS['bw_hz'], description='BW (Hz)')
w_nf = widgets.FloatText(value=DEFAULTS['nf_db'], description='NF (dB)')
w_misc = widgets.FloatText(value=DEFAULTS['misc_loss_db'], description='Lmisc (dB)')
w_fading = widgets.Dropdown(options=['none', 'slow', 'fast'], value=DEFAULTS['fading_mode'], description='Fading')

w_nsub = widgets.IntSlider(value=DEFAULTS['Nsub'], min=16, max=256, step=16, description='Nsub')
w_cp = widgets.IntSlider(value=DEFAULTS['cp_len'], min=0, max=64, step=1, description='CP')
w_M = widgets.Dropdown(options=[4, 16, 64], value=DEFAULTS['M'], description='M-QAM')
w_frames = widgets.IntSlider(value=DEFAULTS['n_frames'], min=5, max=100, step=5, description='Frames')
w_fec = widgets.Checkbox(value=DEFAULTS['fec_enabled'], description='FEC ON')
w_fecn = widgets.IntText(value=DEFAULTS['fec_n'], description='FEC n')
w_feck = widgets.IntText(value=DEFAULTS['fec_k'], description='FEC k')

start_btn = widgets.Button(description='Start sim')
stop_btn = widgets.Button(description='Stop sim')
status = widgets.HTML('<b>Status:</b> idle')
info_box = widgets.HTML()
plot_widget = widgets.Image(format='png', layout=widgets.Layout(width='420px'))
snr_plot_widget = widgets.Image(format='png', layout=widgets.Layout(width='460px'))
ber_plot_widget = widgets.Image(format='png', layout=widgets.Layout(width='460px'))
availability_plot_widget = widgets.Image(format='png', layout=widgets.Layout(width='460px'))

sim_state = {
    'timer': None,
    'counter': 0,
    'running': False,
    'iter_hist': [],
    'snr_hist': [],
    'ber_hist': [],
    'packet_ok_window': [],
    'avail_iter_hist': [],
    'avail_hist': [],
}


def sync_gui(*args):
    manual = w_pl_mode.value == 'manual'
    w_loss.disabled = not manual
    w_dist.disabled = manual
    w_freq.disabled = manual
    fec_on = bool(w_fec.value)
    w_fecn.disabled = not fec_on
    w_feck.disabled = not fec_on


def current_params():
    return {
        'ptx_dbm': w_ptx.value,
        'gtx_dbi': w_gtx.value,
        'grx_dbi': w_grx.value,
        'packet_length_bytes': w_pkt.value,
        'path_loss_mode': w_pl_mode.value,
        'link_loss_db': w_loss.value,
        'distance_m': w_dist.value,
        'freq_mhz': w_freq.value,
        'bw_hz': w_bw.value,
        'nf_db': w_nf.value,
        'misc_loss_db': w_misc.value,
        'fading_mode': w_fading.value,
        'Nsub': w_nsub.value,
        'cp_len': min(w_cp.value, w_nsub.value - 1),
        'M': w_M.value,
        'n_frames': w_frames.value,
        'fec_enabled': w_fec.value,
        'fec_n': w_fecn.value,
        'fec_k': w_feck.value,
        'n_taps': DEFAULTS['n_taps'],
        'availability_window': DEFAULTS['availability_window'],
    }


def reset_history():
    sim_state['counter'] = 0
    sim_state['iter_hist'] = []
    sim_state['snr_hist'] = []
    sim_state['ber_hist'] = []
    sim_state['packet_ok_window'] = []
    sim_state['avail_iter_hist'] = []
    sim_state['avail_hist'] = []


def push_history(iteration, snr_db, ber_value, packet_ok, availability_window):
    sim_state['iter_hist'].append(iteration)
    sim_state['snr_hist'].append(float(snr_db))
    sim_state['ber_hist'].append(float(ber_value))
    sim_state['packet_ok_window'].append(1.0 if packet_ok else 0.0)
    sim_state['iter_hist'] = sim_state['iter_hist'][-12:]
    sim_state['snr_hist'] = sim_state['snr_hist'][-12:]
    sim_state['ber_hist'] = sim_state['ber_hist'][-12:]
    if iteration % availability_window == 0 and sim_state['packet_ok_window']:
        availability_pct = 100.0 * float(np.mean(sim_state['packet_ok_window']))
        sim_state['avail_iter_hist'].append(iteration)
        sim_state['avail_hist'].append(availability_pct)
        sim_state['packet_ok_window'] = []


async def _async_loop():
    while sim_state['running']:
        try:
            run_one_step()
        except Exception as exc:
            sim_state['running'] = False
            status.value = f"<b>Status:</b> error | {type(exc).__name__}: {exc}"
            return
        await asyncio.sleep(1.0)


def schedule_next():
    # kept for compatibility; not used in async mode
    if not sim_state['running'] or IN_COLAB:
        return
    timer = threading.Timer(1.0, tick)
    timer.daemon = True
    sim_state['timer'] = timer
    timer.start()


def run_one_step():
    params = current_params()
    params['seed'] = 1234 + sim_state['counter']
    sim_state['counter'] += 1
    iteration = sim_state['counter']
    result = run_link_demo(**params)
    push_history(
        iteration,
        result.get('instant_snr_db', result['snr_db']),
        result['digital']['ber_with_eq'],
        result['packet_received_ok'],
        params['availability_window'],
    )
    image_bytes, info_html = render_link_demo(result, iteration=iteration)
    snr_bytes, ber_bytes, availability_bytes = render_history_plots(
        sim_state['iter_hist'],
        sim_state['snr_hist'],
        sim_state['ber_hist'],
        sim_state['avail_iter_hist'],
        sim_state['avail_hist'],
        params['availability_window'],
    )
    plot_widget.value = b''
    plot_widget.value = image_bytes
    snr_plot_widget.value = b''
    snr_plot_widget.value = snr_bytes
    ber_plot_widget.value = b''
    ber_plot_widget.value = ber_bytes
    availability_plot_widget.value = b''
    availability_plot_widget.value = availability_bytes
    info_box.value = info_html
    status.value = f"<b>Status:</b> running | iter={iteration}"


def tick():
    if not sim_state['running']:
        return
    try:
        run_one_step()
    except Exception as exc:
        sim_state['running'] = False
        status.value = f"<b>Status:</b> error | {type(exc).__name__}: {exc}"
        return
    schedule_next()


def _colab_tick():
    if not sim_state['running']:
        return None
    try:
        run_one_step()
    except Exception as exc:
        sim_state['running'] = False
        status.value = f"<b>Status:</b> error | {type(exc).__name__}: {exc}"
    return None


if IN_COLAB:
    colab_output.register_callback('link_balance.tick', _colab_tick)


def on_start(_):
    if sim_state['running']:
        return
    reset_history()
    sim_state['running'] = True
    if IN_COLAB:
        _colab_tick()
        colab_output.eval_js("""
            if (window._linkBalanceTimer) { clearInterval(window._linkBalanceTimer); }
            window._linkBalanceTimer = setInterval(() => {
                google.colab.kernel.invokeFunction('link_balance.tick', [], {});
            }, 1000);
        """)
    else:
        asyncio.ensure_future(_async_loop())


def on_stop(_):
    sim_state['running'] = False
    if IN_COLAB:
        try:
            colab_output.eval_js("""
                if (window._linkBalanceTimer) {
                    clearInterval(window._linkBalanceTimer);
                    window._linkBalanceTimer = null;
                }
            """)
        except Exception:
            pass
    # cancel any lingering threading.Timer (fallback path)
    timer = sim_state.get('timer')
    if timer is not None:
        timer.cancel()
        sim_state['timer'] = None
    status.value = '<b>Status:</b> stopped'


w_pl_mode.observe(sync_gui, names='value')
w_fec.observe(sync_gui, names='value')
w_fading.observe(sync_gui, names='value')
sync_gui()

start_btn.on_click(on_start)
stop_btn.on_click(on_stop)

GUI_BOX = widgets.VBox([
    widgets.HTML('<b>Parametros del sistema</b>'),
    w_ptx, w_gtx, w_grx, w_pkt,
    widgets.HTML('<b>Parametros del canal</b>'),
    w_pl_mode, w_loss, w_dist, w_freq, w_bw, w_nf, w_misc, w_fading,
    widgets.HTML('<b>Parametros OFDM</b>'),
    w_nsub, w_cp, w_M, w_frames, w_fec, w_fecn, w_feck,
    widgets.HBox([start_btn, stop_btn]),
    status,
    info_box,
    widgets.HBox([plot_widget, widgets.VBox([snr_plot_widget, ber_plot_widget, availability_plot_widget])]),
])

result_default = run_link_demo(**DEFAULTS)
reset_history()
push_history(
    0,
    result_default.get('instant_snr_db', result_default['snr_db']),
    result_default['digital']['ber_with_eq'],
    result_default['packet_received_ok'],
    DEFAULTS['availability_window'],
)
image_bytes, info_html = render_link_demo(result_default, iteration=0)
snr_bytes, ber_bytes, availability_bytes = render_history_plots(
    sim_state['iter_hist'],
    sim_state['snr_hist'],
    sim_state['ber_hist'],
    sim_state['avail_iter_hist'],
    sim_state['avail_hist'],
    DEFAULTS['availability_window'],
)
plot_widget.value = image_bytes
snr_plot_widget.value = snr_bytes
ber_plot_widget.value = ber_bytes
availability_plot_widget.value = availability_bytes
info_box.value = info_html

display(GUI_BOX)
