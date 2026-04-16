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

PACKAGE_ROOT = PROJECT_ROOT / ("Engine" if IN_COLAB else "src")

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import time
import traceback
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from smartgrids_sim.scenarios import run_digital_scenario


DEFAULTS = {
    'Nsub': 128,
    'cp_len': 16,
    'modulation': 'QAM',
    'M': 16,
    'n_frames': 20,
    'fec_enabled': False,
    'fec_scheme': 'repetition',
    'fec_n': 3,
    'fec_k': 1,
    'snr_db_ini': 0.0,
    'snr_db_fin': 30.0,
    'snr_db_step': 2.0,
    'channel_type': 'awgn',
    'n_taps': 2,
    'rayleigh_severity': 0.35,
    'fs_mhz': 20.0,
}


def compute_net_bitrate(nsub, m, cp_len, fs_hz, fec_enabled=False, fec_n=1, fec_k=1):
    bits_per_symbol = np.log2(m)
    tsym = nsub / fs_hz
    eta_cp = nsub / (nsub + cp_len)
    eta_fec = (fec_k / fec_n) if fec_enabled else 1.0
    return (nsub * bits_per_symbol / tsym) * eta_cp * eta_fec


def run_snr_sweep(params, progress_cb=None):
    nsub = int(params['Nsub'])
    cp_len = min(int(params['cp_len']), nsub - 1)
    snr_ini = float(params['snr_db_ini'])
    snr_fin = float(params['snr_db_fin'])
    snr_step = float(params['snr_db_step'])
    if snr_step <= 0:
        raise ValueError('SNR step debe ser > 0.')
    if snr_fin < snr_ini:
        snr_ini, snr_fin = snr_fin, snr_ini
    fec_scheme = str(params.get('fec_scheme', 'repetition')).lower()

    if int(params['fec_k']) <= 0:
        raise ValueError('fec_k debe ser > 0.')
    if int(params['fec_n']) < int(params['fec_k']):
        raise ValueError('fec_n debe ser >= fec_k.')
    if fec_scheme == 'repetition' and int(params['fec_n']) % int(params['fec_k']) != 0:
        raise ValueError('Para este FEC de repeticion se requiere fec_n multiplo de fec_k.')

    snr_vec = np.arange(snr_ini, snr_fin + 1e-12, snr_step)
    if snr_vec.size == 0:
        raise ValueError('Rango SNR vacio.')

    ber_no_eq_no_fec = []
    ber_eq_no_fec = []
    ber_eq_fec = []

    start_time = time.time()
    for i, snr_i in enumerate(snr_vec):
        if progress_cb is not None:
            progress_cb(i, len(snr_vec), snr_i)

        point_seed = 700 + i
        r_no_fec = run_digital_scenario(
            Nsub=nsub,
            cp_len=cp_len,
            M=int(params['M']),
            snr_db=float(snr_i),
            n_frames=max(8, int(params['n_frames']) // 2),
            fec_enabled=False,
            fec_n=int(params['fec_n']),
            fec_k=int(params['fec_k']),
            channel_type=str(params['channel_type']),
            n_taps=int(params['n_taps']),
            rayleigh_severity=float(params.get('rayleigh_severity', 1.0)),
            seed=point_seed,
        )
        ber_no_eq_no_fec.append(r_no_fec['ber_without_eq'])
        ber_eq_no_fec.append(r_no_fec['ber_with_eq'])

        r_fec = run_digital_scenario(
            Nsub=nsub,
            cp_len=cp_len,
            M=int(params['M']),
            snr_db=float(snr_i),
            n_frames=max(8, int(params['n_frames']) // 2),
            fec_enabled=True,
            fec_scheme=fec_scheme,
            fec_n=int(params['fec_n']),
            fec_k=int(params['fec_k']),
            channel_type=str(params['channel_type']),
            n_taps=int(params['n_taps']),
            rayleigh_severity=float(params.get('rayleigh_severity', 1.0)),
            seed=point_seed,
        )
        ber_eq_fec.append(r_fec['ber_with_eq'])

    ber_no_eq_no_fec = np.asarray(ber_no_eq_no_fec, dtype=float)
    ber_eq_no_fec = np.asarray(ber_eq_no_fec, dtype=float)
    ber_eq_fec = np.asarray(ber_eq_fec, dtype=float)

    ber_no_eq_no_fec[ber_no_eq_no_fec == 0] = np.nan
    ber_eq_no_fec[ber_eq_no_fec == 0] = np.nan
    ber_eq_fec[ber_eq_fec == 0] = np.nan

    mid_snr_db = 0.5 * (snr_ini + snr_fin)
    constellation_run = run_digital_scenario(
        Nsub=nsub,
        cp_len=cp_len,
        M=int(params['M']),
        snr_db=float(mid_snr_db),
        n_frames=max(8, int(params['n_frames']) // 2),
        fec_enabled=bool(params['fec_enabled']),
        fec_scheme=fec_scheme,
        fec_n=int(params['fec_n']),
        fec_k=int(params['fec_k']),
        channel_type=str(params['channel_type']),
        n_taps=int(params['n_taps']),
        rayleigh_severity=float(params.get('rayleigh_severity', 1.0)),
        seed=1200 + len(snr_vec),
    )

    return {
        'snr_vec': snr_vec,
        'ber_no_eq_no_fec': ber_no_eq_no_fec,
        'ber_eq_no_fec': ber_eq_no_fec,
        'ber_eq_fec': ber_eq_fec,
        'constellation_snr_db': mid_snr_db,
        'constellation_rx_symbols_pre_eq': np.asarray(constellation_run['rx_symbols_pre_eq'], dtype=np.complex128),
        'constellation_rx_symbols_post_eq': np.asarray(constellation_run['rx_symbols_post_eq'], dtype=np.complex128),
        'constellation_tx_symbols': np.asarray(constellation_run['tx_symbols'], dtype=np.complex128),
        'elapsed_s': time.time() - start_time,
    }


_existing_gui_box = globals().get('GUI_BOX', None)
if _existing_gui_box is not None:
    try:
        _existing_gui_box.close()
    except Exception:
        pass

style_html = HTML("""
<style>
.gui-root {
    font-family: Arial, sans-serif;
    color: #f0f0f0;
}
.gui-title {
    color: #f0f0f0;
    font-weight: 600;
    margin: 0 0 6px 0;
}
.gui-subtitle {
    color: #f0f0f0;
    font-weight: 600;
    margin: 0 0 6px 0;
}
</style>
""")
display(style_html)

BOX_BG = '#1e1e1e'
PANEL_BG = '#222222'
BORDER = '1px solid #6b6b6b'
TEXT = '#f0f0f0'

w_nsub = widgets.BoundedIntText(value=DEFAULTS['Nsub'], min=16, max=4096, step=1, description='Ncp')
w_cp = widgets.BoundedIntText(value=DEFAULTS['cp_len'], min=0, max=1024, step=1, description='CP Length')
w_mod = widgets.Dropdown(options=['QAM'], value=DEFAULTS['modulation'], description='Modulation')
w_m = widgets.Dropdown(options=[4, 16, 64], value=DEFAULTS['M'], description='Mod. Order')
w_frames = widgets.BoundedIntText(value=DEFAULTS['n_frames'], min=1, max=2000, step=1, description='Frames')
w_fec = widgets.Checkbox(value=DEFAULTS['fec_enabled'], description='FEC ON')
w_fec_type = widgets.RadioButtons(
    options=[('RC (Repetition Code)', 'repetition'), ('Conv. Enc. (Viterbi)', 'convolutional')],
    value=DEFAULTS['fec_scheme'],
    description='FEC type',
)
w_fec_n = widgets.BoundedIntText(value=DEFAULTS['fec_n'], min=1, max=64, step=1, description='FEC n')
w_fec_k = widgets.BoundedIntText(value=DEFAULTS['fec_k'], min=1, max=64, step=1, description='FEC k')
w_rc = widgets.HTMLMath()
w_txrate = widgets.HTMLMath()

w_snr_ini = widgets.FloatText(value=DEFAULTS['snr_db_ini'], description='SNR min (dB)')
w_snr_fin = widgets.FloatText(value=DEFAULTS['snr_db_fin'], description='SNR max(dB)')
w_snr_step = widgets.FloatText(value=DEFAULTS['snr_db_step'], description='SNR step(dB)')
w_channel = widgets.Dropdown(options=['rayleigh', 'awgn'], value=DEFAULTS['channel_type'], description='Canal')
w_ntaps = widgets.BoundedIntText(value=DEFAULTS['n_taps'], min=1, max=64, step=1, description='n_taps')
w_fs = widgets.FloatText(value=DEFAULTS['fs_mhz'], description='Fs (MHz)') 

run_button = widgets.Button(description='Ejecuta Simulacion', button_style='')
status = widgets.HTML('<span style="color:#f0f0f0;">Listo.</span>')
metrics_html = widgets.HTML()
out_plot = widgets.Output(layout=widgets.Layout(border=BORDER, height='430px'))
out_log = widgets.Output(layout=widgets.Layout(border=BORDER, height='120px', overflow='auto'))

for w in [
    w_nsub,
    w_cp,
    w_mod,
    w_m,
    w_frames,
    w_fec,
    w_fec_type,
    w_fec_n,
    w_fec_k,
    w_snr_ini,
    w_snr_fin,
    w_snr_step,
    w_channel,
    w_ntaps,
    w_fs,
]:
    try:
        w.style.description_width = '110px'
    except Exception:
        pass
    w.layout = widgets.Layout(width='100%')

run_button.layout = widgets.Layout(width='100%')
status.layout = widgets.Layout(width='100%')
metrics_html.layout = widgets.Layout(width='100%')
w_rc.layout = widgets.Layout(width='100%')
w_txrate.layout = widgets.Layout(width='100%')


def _update_rc_widget(*args):
    fec_enabled = w_fec.value
    if fec_enabled and w_fec_type.value == 'convolutional':
        rc = 0.5
        fec_n_eff, fec_k_eff = 2, 1
    elif fec_enabled:
        rc = float(w_fec_k.value) / float(w_fec_n.value)
        fec_n_eff, fec_k_eff = int(w_fec_n.value), int(w_fec_k.value)
    else:
        rc = 1.0
        fec_n_eff, fec_k_eff = int(w_fec_n.value), int(w_fec_k.value)

    w_rc.value = rf'R_c = {rc:.3f}'
    tx_rate = compute_net_bitrate(
        nsub=int(w_nsub.value),
        m=int(w_m.value),
        cp_len=int(w_cp.value),
        fs_hz=float(w_fs.value) * 1e6,
        fec_enabled=fec_enabled,
        fec_n=fec_n_eff,
        fec_k=fec_k_eff,
    )
    w_txrate.value = rf'Vel. Tx = {tx_rate/1e6:.3f} Mbps'


def _sync_fec_widgets(*args):
    enabled = w_fec.value
    scheme = w_fec_type.value
    w_fec_type.disabled = not enabled
    use_rep_params = enabled and scheme == 'repetition'
    w_fec_n.disabled = not use_rep_params
    w_fec_k.disabled = not use_rep_params
    _update_rc_widget()


w_fec.observe(_sync_fec_widgets, names='value')
w_fec_type.observe(_sync_fec_widgets, names='value')
w_fec_n.observe(_update_rc_widget, names='value')
w_fec_k.observe(_update_rc_widget, names='value')
w_nsub.observe(_update_rc_widget, names='value')
w_cp.observe(_update_rc_widget, names='value')
w_m.observe(_update_rc_widget, names='value')
w_fs.observe(_update_rc_widget, names='value')
_update_rc_widget()

panel_tx = widgets.VBox([
    widgets.HTML('<div class="gui-subtitle">Parametros Tx</div>'),
    w_nsub, w_cp, w_mod, w_m, w_frames, w_fs, w_fec, w_fec_type, w_fec_n, w_fec_k, w_rc, w_txrate
], layout=widgets.Layout(border=BORDER, padding='10px', background_color=BOX_BG))

panel_ch = widgets.VBox([
    widgets.HTML('<div class="gui-subtitle">Parametros Canal</div>'),
    w_snr_ini, w_snr_fin, w_snr_step, w_channel
], layout=widgets.Layout(border=BORDER, padding='10px', background_color=BOX_BG))

panel_exec = widgets.VBox([
    widgets.HTML('<div class="gui-subtitle">Ejecucion</div>'),
    run_button, status, metrics_html
], layout=widgets.Layout(border=BORDER, padding='10px', background_color=BOX_BG))

left_col = widgets.VBox([
    widgets.HTML('<div class="gui-title">Simulacion</div>'),
    panel_tx, panel_ch, panel_exec
], layout=widgets.Layout(width='270px', border=BORDER, padding='10px', background_color=BOX_BG, margin='0 10px 0 0'))

plot_panel = widgets.VBox([
    widgets.HTML('<div class="gui-title">Resultados</div>'),
    out_plot,
    out_log,
], layout=widgets.Layout(flex='1 1 auto', border=BORDER, padding='10px', background_color=BOX_BG))

GUI_BOX = widgets.HBox([left_col, plot_panel], layout=widgets.Layout(width='100%', align_items='stretch'))


def _append_log(line):
    with out_log:
        print(line)


def _progress_cb(i, total, snr_i):
    status.value = f'<span style="color:#f0f0f0;">Simulando SNR = {snr_i:.1f} dB ({i+1}/{total})...</span>'


def _plot_valid_semilogy(ax, x, y, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if np.any(mask):
        ax.semilogy(x[mask], y[mask], marker='o', label=label)


def _sample_symbols(symbols, max_points=1500):
    symbols = np.asarray(symbols, dtype=np.complex128).ravel()
    if symbols.size <= max_points:
        return symbols
    idx = np.linspace(0, symbols.size - 1, max_points, dtype=int)
    return symbols[idx]


def _plot_constellation(ax, rx_symbols, tx_symbols, snr_db, title_suffix):
    rx_symbols = _sample_symbols(rx_symbols)
    tx_symbols = _sample_symbols(np.unique(np.asarray(tx_symbols, dtype=np.complex128)))

    if rx_symbols.size:
        ax.scatter(
            rx_symbols.real,
            rx_symbols.imag,
            s=10,
            alpha=0.35,
            color='#7bdff2',
            edgecolors='none',
            label=title_suffix,
        )

    if tx_symbols.size:
        ax.scatter(
            tx_symbols.real,
            tx_symbols.imag,
            s=60,
            color='#ff9f1c',
            marker='x',
            linewidths=1.5,
            label='Puntos ideales',
        )

    all_symbols = np.concatenate([arr for arr in [rx_symbols, tx_symbols] if arr.size], dtype=np.complex128)
    if all_symbols.size:
        lim = 1.15 * np.max(np.abs(np.concatenate([all_symbols.real, all_symbols.imag])))
        lim = max(lim, 1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_title(f'{title_suffix} a SNR medio = {snr_db:.1f} dB')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')


def _update_metrics(params, results=None):
    fec_scheme = str(params.get('fec_scheme', 'repetition')).lower()
    if params['fec_enabled'] and fec_scheme == 'convolutional':
        fec_n_eff, fec_k_eff = 2, 1
        fec_label = 'Conv. Enc. 1/2 (Viterbi)'
    else:
        fec_n_eff = int(params['fec_n'])
        fec_k_eff = int(params['fec_k'])
        fec_label = 'RC (Repetition)' if params['fec_enabled'] else 'None'

    bitrate = compute_net_bitrate(
        nsub=int(params['Nsub']),
        m=int(params['M']),
        cp_len=int(params['cp_len']),
        fs_hz=float(params['fs_hz']),
        fec_enabled=bool(params['fec_enabled']),
        fec_n=fec_n_eff,
        fec_k=fec_k_eff,
    )
    extra = ''
    if results is not None:
        extra = f"<br><span style='color:#d0d0d0;'>Tiempo: {results['elapsed_s']:.1f} s</span>"
    metrics_html.value = (
        f"<div style='color:#f0f0f0; line-height:1.5;'>"
        f"Bitrate neto: <b>{bitrate/1e6:.2f} Mbps</b>"
        f"<br>Canal: <b>{params['channel_type']}</b>"
        f"<br>Modulacion: <b>{params['M']}-QAM</b>"
        f"<br>FEC type: <b>{fec_label}</b>"
        f"<br>FEC: <b>{fec_k_eff}/{fec_n_eff if params['fec_enabled'] else 1}</b>"
        f"<br>R<sub>c</sub>: <b>{(fec_k_eff / fec_n_eff) if params['fec_enabled'] else 1.0:.3f}</b>"
        f"{extra}</div>"
    )


def _on_run(_):
    with out_plot:
        clear_output(wait=True)
    with out_log:
        clear_output(wait=True)

    params = {
        'Nsub': int(w_nsub.value),
        'cp_len': int(w_cp.value),
        'modulation': w_mod.value,
        'M': int(w_m.value),
        'n_frames': int(w_frames.value),
        'fec_enabled': bool(w_fec.value),
        'fec_scheme': str(w_fec_type.value),
        'fec_n': int(w_fec_n.value),
        'fec_k': int(w_fec_k.value),
        'snr_db_ini': float(w_snr_ini.value),
        'snr_db_fin': float(w_snr_fin.value),
        'snr_db_step': float(w_snr_step.value),
        'channel_type': str(w_channel.value),
        'n_taps': int(w_ntaps.value),
        'rayleigh_severity': float(DEFAULTS['rayleigh_severity']),
        'fs_hz': float(w_fs.value) * 1e6,
    }

    if params['cp_len'] >= params['Nsub']:
        params['cp_len'] = params['Nsub'] - 1
        w_cp.value = params['cp_len']
        _append_log(f"CP ajustado automaticamente a {params['cp_len']}.")

    _update_metrics(params)

    run_button.disabled = True
    try:
        _append_log('Arrancando simulacion...')
        results = run_snr_sweep(params, progress_cb=_progress_cb)

        with out_plot:
            clear_output(wait=True)
            plt.close('all')
            with plt.style.context('dark_background'):
                fig, (ax_ber, ax_const_pre, ax_const_post) = plt.subplots(
                    1, 3, figsize=(17.0, 5.2), gridspec_kw={'width_ratios': [1.7, 1.0, 1.0]}
                )
                _plot_valid_semilogy(ax_ber, results['snr_vec'], results['ber_no_eq_no_fec'], 'Sin EQ, sin FEC')
                _plot_valid_semilogy(ax_ber, results['snr_vec'], results['ber_eq_no_fec'], 'Con EQ, sin FEC')
                _plot_valid_semilogy(ax_ber, results['snr_vec'], results['ber_eq_fec'], 'Con EQ, con FEC')
                ax_ber.set_xlabel('SNR (dB)')
                ax_ber.set_ylabel('BER')
                ax_ber.set_title('BER vs SNR')
                ax_ber.grid(True, which='both', alpha=0.35)
                ax_ber.legend(loc='best')
                _plot_constellation(
                    ax_const_pre,
                    results['constellation_rx_symbols_pre_eq'],
                    results['constellation_tx_symbols'],
                    results['constellation_snr_db'],
                    'Constelacion pre-EQ',
                )
                _plot_constellation(
                    ax_const_post,
                    results['constellation_rx_symbols_post_eq'],
                    results['constellation_tx_symbols'],
                    results['constellation_snr_db'],
                    'Constelacion post-EQ',
                )
                fig.tight_layout()
                display(fig)
                plt.close(fig)

        _append_log('Simulacion completada correctamente.')
        _append_log(f"SNR barrido: {results['snr_vec'][0]:.1f} a {results['snr_vec'][-1]:.1f} dB en {len(results['snr_vec'])} puntos.")
        _append_log(f"Constelaciones pre/post-EQ mostradas para SNR medio = {results['constellation_snr_db']:.1f} dB.")
        _append_log(f"BER final con EQ: {np.nanmin(results['ber_eq_no_fec']):.3e}")
        status.value = '<span style="color:#f0f0f0;">Simulacion completada.</span>'
        _update_metrics(params, results)
    except Exception as e:
        status.value = f'<span style="color:#ff8080;">Error: {type(e).__name__}: {e}</span>'
        _append_log('ERROR durante la simulacion:')
        _append_log(traceback.format_exc())
    finally:
        run_button.disabled = False


try:
    run_button._click_handlers.callbacks.clear()
except Exception:
    pass
run_button.on_click(_on_run)
_update_metrics({
    'Nsub': w_nsub.value,
    'cp_len': w_cp.value,
    'M': w_m.value,
    'fec_enabled': bool(w_fec.value),
    'fec_scheme': str(w_fec_type.value),
    'fec_n': w_fec_n.value,
    'fec_k': w_fec_k.value,
    'channel_type': w_channel.value,
    'fs_hz': float(w_fs.value) * 1e6,
})
display(GUI_BOX)
