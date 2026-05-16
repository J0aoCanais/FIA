"""
plot_results.py — Gráficos para o relatório do Lunar Lander (TP2 FIA).

Uso:
  python plot_results.py                          # tudo, usa ./resultados e ./resultados2
  python plot_results.py --dir1 ../resultados/    # sem vento noutro sítio
  python plot_results.py --dir2 ../resultados2/   # com vento noutro sítio
  python plot_results.py --out ../graficos/        # pasta de saída
  python plot_results.py --show                   # abre janelas em vez de guardar PDF
  python plot_results.py --no-wind                # só sem vento

Gráficos gerados:
  fig1_all_experiments.pdf   — painel 4×2 com curvas de evolução (fitness vs geração)
  fig2_comparison.pdf        — todas as 8 experiências num único plot (médias)
  fig3_factors.pdf           — análise por fator (mutação / crossover / elitismo)
  fig4_success_bar.pdf       — taxa de sucesso por configuração (requer test_results.csv)
  fig5_wind_comparison.pdf   — sem vento vs com vento (requer ambos os dirs)
  fig6_summary_table.pdf     — tabela de resultados (fitness + sucesso)

test_results.csv (opcional):
  Gerado pelo script principal (MODE='test_all'). Formato:
  exp,run,mean_fitness,success_rate
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ============================================================
# Configuração das 8 experiências (Tabela 2 do enunciado)
# ============================================================
EXPERIMENTS = {
    1: {'mut': 0.008, 'cx': 0.5, 'elite': 0},
    2: {'mut': 0.05,  'cx': 0.5, 'elite': 0},
    3: {'mut': 0.008, 'cx': 0.9, 'elite': 0},
    4: {'mut': 0.05,  'cx': 0.9, 'elite': 0},
    5: {'mut': 0.008, 'cx': 0.5, 'elite': 1},
    6: {'mut': 0.05,  'cx': 0.5, 'elite': 1},
    7: {'mut': 0.008, 'cx': 0.9, 'elite': 1},
    8: {'mut': 0.05,  'cx': 0.9, 'elite': 1},
}

COLORS = plt.cm.tab10(np.linspace(0, 0.8, 8))


def short_label(exp_num):
    c = EXPERIMENTS[exp_num]
    return f"E{exp_num}\nmut={c['mut']}\ncx={c['cx']}\nelite={c['elite']}"


def long_label(exp_num):
    c = EXPERIMENTS[exp_num]
    return f"Exp {exp_num}: mut={c['mut']}, cx={c['cx']}, elite={c['elite']}"


# ============================================================
# Leitura dos logs
# ============================================================
def read_log(filepath):
    fitnesses = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                fitnesses.append(float(line.split('\t')[0]))
    return np.array(fitnesses)


def load_experiment(exp_num, log_dir, n_runs=5):
    """Carrega os N runs de uma experiência. Retorna array (n_runs, n_gens) ou None."""
    runs = []
    for run in range(n_runs):
        path = os.path.join(log_dir, f'log_e{exp_num}_r{run}.txt')
        if os.path.exists(path):
            runs.append(read_log(path))
    if not runs:
        return None
    min_len = min(len(r) for r in runs)
    return np.array([r[:min_len] for r in runs])


def load_all(log_dir, n_runs=5):
    return {e: load_experiment(e, log_dir, n_runs) for e in EXPERIMENTS}


# ============================================================
# Leitura de resultados de teste (CSV)
# ============================================================
def load_test_results(csv_path):
    """
    Lê test_results.csv com colunas: exp,run,mean_fitness,success_rate
    Devolve dict {exp_num: {'fitness': [...], 'success': [...]}}
    """
    results = {e: {'fitness': [], 'success': []} for e in EXPERIMENTS}
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(row['exp'])
            results[e]['fitness'].append(float(row['mean_fitness']))
            results[e]['success'].append(float(row['success_rate']))
    # Só devolve se tiver dados
    if all(len(v['fitness']) == 0 for v in results.values()):
        return None
    return results


# ============================================================
# FIG 1 — Painel 4×2: evolução por experiência
# ============================================================
def fig1_all_experiments(all_data, save_dir, show=False, label=''):
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True)
    axes = axes.flatten()

    for idx, exp_num in enumerate(sorted(all_data)):
        ax = axes[idx]
        data = all_data[exp_num]
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        gens = np.arange(data.shape[1])
        mean = data.mean(axis=0)
        std  = data.std(axis=0)

        for i, run in enumerate(data):
            ax.plot(gens, run, alpha=0.25, linewidth=0.8, color=COLORS[idx])
        ax.plot(gens, mean, color=COLORS[idx], linewidth=2.2, label='Média')
        ax.fill_between(gens, mean - std, mean + std,
                        color=COLORS[idx], alpha=0.2, label='±1σ')

        c = EXPERIMENTS[exp_num]
        ax.set_title(
            f"Exp {exp_num} — mut={c['mut']}, cx={c['cx']}, elite={c['elite']}",
            fontsize=10, fontweight='bold'
        )
        ax.set_ylabel('Fitness', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    for ax in axes[-2:]:
        ax.set_xlabel('Geração', fontsize=9)

    title = f'Evolução do Fitness por Geração — 8 Experiências'
    if label:
        title += f' ({label})'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    fname = os.path.join(save_dir, f'fig1_all_experiments{"_" + label if label else ""}.pdf')
    _save_or_show(fig, fname, show)


# ============================================================
# FIG 2 — Comparação das 8 experiências (médias)
# ============================================================
def fig2_comparison(all_data, save_dir, show=False, label=''):
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, exp_num in enumerate(sorted(all_data)):
        data = all_data[exp_num]
        if data is None or len(data) == 0:
            continue
        gens = np.arange(data.shape[1])
        mean = data.mean(axis=0)
        std  = data.std(axis=0)
        ax.plot(gens, mean, color=COLORS[idx], linewidth=1.8,
                label=long_label(exp_num))
        ax.fill_between(gens, mean - std, mean + std,
                        color=COLORS[idx], alpha=0.1)

    title = 'Comparação das 8 Experiências — Fitness Médio (±1σ)'
    if label:
        title += f' [{label}]'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Geração')
    ax.set_ylabel('Fitness médio (melhor indivíduo)')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = os.path.join(save_dir, f'fig2_comparison{"_" + label if label else ""}.pdf')
    _save_or_show(fig, fname, show)


# ============================================================
# FIG 3 — Análise por fator (mutação / crossover / elitismo)
# ============================================================
def fig3_factors(all_data, save_dir, show=False, label=''):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Pares de experiências para cada fator (fixando os outros dois)
    factor_pairs = [
        # (fator, título do eixo-x, grupos [(exp_a, exp_b, label_a, label_b, título)])
        ('Mutação', [
            (1, 2, 'mut=0.008', 'mut=0.05', 'cx=0.5, elite=0'),
            (3, 4, 'mut=0.008', 'mut=0.05', 'cx=0.9, elite=0'),
            (5, 6, 'mut=0.008', 'mut=0.05', 'cx=0.5, elite=1'),
            (7, 8, 'mut=0.008', 'mut=0.05', 'cx=0.9, elite=1'),
        ]),
        ('Crossover', [
            (1, 3, 'cx=0.5', 'cx=0.9', 'mut=0.008, elite=0'),
            (2, 4, 'cx=0.5', 'cx=0.9', 'mut=0.05,  elite=0'),
            (5, 7, 'cx=0.5', 'cx=0.9', 'mut=0.008, elite=1'),
            (6, 8, 'cx=0.5', 'cx=0.9', 'mut=0.05,  elite=1'),
        ]),
        ('Elitismo', [
            (1, 5, 'elite=0', 'elite=1', 'mut=0.008, cx=0.5'),
            (2, 6, 'elite=0', 'elite=1', 'mut=0.05,  cx=0.5'),
            (3, 7, 'elite=0', 'elite=1', 'mut=0.008, cx=0.9'),
            (4, 8, 'elite=0', 'elite=1', 'mut=0.05,  cx=0.9'),
        ]),
    ]

    linestyles = ['-', '--', '-.', ':']
    low_colors  = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b']
    high_colors = ['#ff7f0e', '#d62728', '#e377c2', '#bcbd22']

    for ax, (factor_name, groups) in zip(axes, factor_pairs):
        legend_handles = []
        for i, (ea, eb, la, lb, subtitle) in enumerate(groups):
            da = all_data.get(ea)
            db = all_data.get(eb)
            ls = linestyles[i]
            if da is not None and len(da) > 0:
                gens = np.arange(da.shape[1])
                ax.plot(gens, da.mean(axis=0),
                        color=low_colors[i], linestyle=ls, linewidth=1.5,
                        label=f'Exp {ea} ({la}) [{subtitle}]')
            if db is not None and len(db) > 0:
                gens = np.arange(db.shape[1])
                ax.plot(gens, db.mean(axis=0),
                        color=high_colors[i], linestyle=ls, linewidth=1.5,
                        label=f'Exp {eb} ({lb}) [{subtitle}]')

        ax.set_title(f'Efeito do {factor_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Geração')
        ax.set_ylabel('Fitness médio')
        ax.legend(fontsize=6.5, loc='lower right')
        ax.grid(True, alpha=0.3)

    title = 'Análise por Fator'
    if label:
        title += f' [{label}]'
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()

    fname = os.path.join(save_dir, f'fig3_factors{"_" + label if label else ""}.pdf')
    _save_or_show(fig, fname, show)


# ============================================================
# FIG 4 — Taxa de sucesso e fitness médio (barras)
# ============================================================
def fig4_success_bar(test_results, save_dir, show=False, label=''):
    """Requer test_results (dict de {exp: {fitness:[], success:[]}}). """
    if test_results is None:
        print(f'  [fig4] Sem test_results.csv — a saltar gráfico de taxa de sucesso.')
        return

    exp_nums = sorted(test_results.keys())
    x = np.arange(len(exp_nums))
    width = 0.35

    mean_success = [np.mean(test_results[e]['success']) * 100 for e in exp_nums]
    std_success  = [np.std(test_results[e]['success'])  * 100 for e in exp_nums]
    mean_fit     = [np.mean(test_results[e]['fitness'])       for e in exp_nums]
    std_fit      = [np.std(test_results[e]['fitness'])        for e in exp_nums]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Taxa de sucesso
    bars = ax1.bar(x, mean_success, yerr=std_success, capsize=5,
                   color=[COLORS[i] for i in range(len(exp_nums))],
                   edgecolor='black', linewidth=0.6, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([short_label(e) for e in exp_nums], fontsize=8)
    ax1.set_ylabel('Taxa de Sucesso (%)')
    ax1.set_ylim(0, 105)
    ax1.set_title('Taxa de Sucesso por Configuração', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val, std in zip(bars, mean_success, std_success):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Fitness médio
    bars2 = ax2.bar(x, mean_fit, yerr=std_fit, capsize=5,
                    color=[COLORS[i] for i in range(len(exp_nums))],
                    edgecolor='black', linewidth=0.6, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([short_label(e) for e in exp_nums], fontsize=8)
    ax2.set_ylabel('Fitness Médio')
    ax2.set_title('Fitness Médio nos Testes por Configuração', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    title = 'Resultados dos Testes (melhor indivíduo, múltiplos episódios)'
    if label:
        title += f' [{label}]'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()

    fname = os.path.join(save_dir, f'fig4_success_bar{"_" + label if label else ""}.pdf')
    _save_or_show(fig, fname, show)


# ============================================================
# FIG 5 — Sem vento vs Com vento
# ============================================================
def fig5_wind_comparison(data_no_wind, data_wind, save_dir, show=False):
    if data_no_wind is None or data_wind is None:
        print('  [fig5] Faltam dados de um dos ambientes — a saltar.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

    for ax, (all_data, title_label) in zip(axes, [
        (data_no_wind, 'Sem Vento'),
        (data_wind,    'Com Vento'),
    ]):
        for idx, exp_num in enumerate(sorted(all_data)):
            data = all_data[exp_num]
            if data is None or len(data) == 0:
                continue
            gens = np.arange(data.shape[1])
            mean = data.mean(axis=0)
            ax.plot(gens, mean, color=COLORS[idx], linewidth=1.6,
                    label=f'Exp {exp_num}')
        ax.set_title(title_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Geração')
        ax.set_ylabel('Fitness médio')
        ax.legend(fontsize=8, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Comparação: Sem Vento vs Com Vento — Evolução do Fitness',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    fname = os.path.join(save_dir, 'fig5_wind_comparison.pdf')
    _save_or_show(fig, fname, show)


# ============================================================
# FIG 6 — Tabela resumo (figura)
# ============================================================
def fig6_summary_table(all_data, test_results, save_dir, show=False, label=''):
    rows = []
    for exp_num in sorted(all_data):
        data = all_data[exp_num]
        c = EXPERIMENTS[exp_num]
        if data is not None and len(data) > 0:
            last = data[:, -1]
            train_mean = f'{last.mean():.2f}'
            train_std  = f'{last.std():.2f}'
        else:
            train_mean = train_std = '—'

        if test_results and exp_num in test_results and test_results[exp_num]['success']:
            s = test_results[exp_num]['success']
            f = test_results[exp_num]['fitness']
            sr   = f'{np.mean(s)*100:.1f}% ± {np.std(s)*100:.1f}%'
            fm   = f'{np.mean(f):.2f} ± {np.std(f):.2f}'
        else:
            sr = fm = '—'

        rows.append([
            str(exp_num),
            str(c['mut']),
            str(c['cx']),
            str(c['elite']),
            f'{train_mean} ± {train_std}',
            fm,
            sr,
        ])

    col_labels = ['Exp', 'Mutação', 'Crossover', 'Elitismo',
                  'Fitness treino\n(última gen, ±σ)',
                  'Fitness teste\n(±σ)',
                  'Taxa sucesso\n(±σ)']

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Cabeçalho colorido
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c5f8a')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Linhas alternadas
    for i in range(1, len(rows) + 1):
        color = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    title = 'Tabela de Resultados das 8 Experiências'
    if label:
        title += f' [{label}]'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    fig.tight_layout()

    fname = os.path.join(save_dir, f'fig6_summary_table{"_" + label if label else ""}.pdf')
    _save_or_show(fig, fname, show)


# ============================================================
# Tabela resumo no terminal
# ============================================================
def print_terminal_table(all_data, test_results=None, label=''):
    print(f'\n{"="*80}')
    if label:
        print(f'  {label}')
    print(f'{"Exp":>4} {"mut":>6} {"cx":>5} {"elite":>6} '
          f'{"Fit. treino (últ. gen)":>24} {"±σ":>7} '
          f'{"Sucesso":>10} {"Fit. teste":>12}')
    print(f'{"-"*80}')
    for exp_num in sorted(all_data):
        data = all_data[exp_num]
        c = EXPERIMENTS[exp_num]
        if data is not None and len(data) > 0:
            last = data[:, -1]
            tm, ts = f'{last.mean():.2f}', f'{last.std():.2f}'
        else:
            tm = ts = '—'
        if test_results and test_results.get(exp_num, {}).get('success'):
            sr = f"{np.mean(test_results[exp_num]['success'])*100:.1f}%"
            fm = f"{np.mean(test_results[exp_num]['fitness']):.2f}"
        else:
            sr = fm = '—'
        print(f'{exp_num:>4} {c["mut"]:>6} {c["cx"]:>5} {c["elite"]:>6} '
              f'{tm:>24} {ts:>7} {sr:>10} {fm:>12}')
    print(f'{"="*80}\n')


# ============================================================
# Utilidade
# ============================================================
def _save_or_show(fig, fname, show):
    if show:
        plt.show()
    else:
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'  Guardado: {fname}')
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1',  default='../resultados/',  help='Logs sem vento')
    parser.add_argument('--dir2',  default='../resultados2/', help='Logs com vento')
    parser.add_argument('--out',   default='../graficos/',    help='Pasta de saída')
    parser.add_argument('--runs',  type=int, default=5)
    parser.add_argument('--show',  action='store_true')
    parser.add_argument('--no-wind', action='store_true', help='Ignora resultados2/')
    parser.add_argument('--csv1',  default=None, help='test_results CSV sem vento')
    parser.add_argument('--csv2',  default=None, help='test_results CSV com vento')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Sem vento ---
    print(f'\nA ler logs sem vento: {os.path.abspath(args.dir1)}')
    data_no_wind = load_all(args.dir1, args.runs) if os.path.isdir(args.dir1) else None
    test1 = load_test_results(args.csv1) if args.csv1 else \
            load_test_results(os.path.join(args.dir1, 'test_results.csv'))

    if data_no_wind:
        print_terminal_table(data_no_wind, test1, label='Sem Vento')
        print('A gerar gráficos sem vento...')
        fig1_all_experiments(data_no_wind, args.out, args.show, label='sem_vento')
        fig2_comparison(data_no_wind, args.out, args.show, label='sem_vento')
        fig3_factors(data_no_wind, args.out, args.show, label='sem_vento')
        fig4_success_bar(test1, args.out, args.show, label='sem_vento')
        fig6_summary_table(data_no_wind, test1, args.out, args.show, label='sem_vento')

    # --- Com vento ---
    data_wind = None
    test2 = None
    if not args.no_wind and os.path.isdir(args.dir2):
        print(f'\nA ler logs com vento: {os.path.abspath(args.dir2)}')
        data_wind = load_all(args.dir2, args.runs)
        test2 = load_test_results(args.csv2) if args.csv2 else \
                load_test_results(os.path.join(args.dir2, 'test_results.csv'))
        if data_wind:
            print_terminal_table(data_wind, test2, label='Com Vento')
            print('A gerar gráficos com vento...')
            fig1_all_experiments(data_wind, args.out, args.show, label='com_vento')
            fig2_comparison(data_wind, args.out, args.show, label='com_vento')
            fig3_factors(data_wind, args.out, args.show, label='com_vento')
            fig4_success_bar(test2, args.out, args.show, label='com_vento')
            fig6_summary_table(data_wind, test2, args.out, args.show, label='com_vento')

    # --- Comparação sem/com vento ---
    if data_no_wind and data_wind:
        print('A gerar comparação sem vento vs com vento...')
        fig5_wind_comparison(data_no_wind, data_wind, args.out, args.show)

    print('\nConcluído.')
    print(f'Gráficos em: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
