"""
gerar_graficos.py — Gera todos os gráficos do TP2 (Fase 1 + Fase 2).

Uso:
    python3 gerar_graficos.py           # guarda PDFs em DIR_GRAFICOS
    python3 gerar_graficos.py --show    # abre janelas em vez de guardar

Gráficos gerados (Fase 1):
    fig1_all_experiments_sem_vento.pdf  — painel 4×2 de curvas por experiência
    fig1_all_experiments_com_vento.pdf
    fig2_comparison_sem_vento.pdf       — 8 experiências num único plot
    fig2_comparison_com_vento.pdf
    fig3_factors_sem_vento.pdf          — análise por fator (mut / cx / elite)
    fig3_factors_com_vento.pdf
    fig4_success_bar_sem_vento.pdf      — taxa de sucesso + fitness médio (barras)
    fig4_success_bar_com_vento.pdf
    fig5_wind_comparison.pdf            — sem vento vs com vento
    fig6_summary_table_sem_vento.pdf    — tabela resumo
    fig6_summary_table_com_vento.pdf

Gráficos gerados (Fase 2):
    fig_p2_curves.pdf                   — curvas das 6 combinações (média ±1σ)
    fig_p2_bars.pdf                     — barras de fitness e taxa de sucesso
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_results as pr

# ============================================================
# CONFIGURAÇÃO — ajustar caminhos conforme necessário
# ============================================================
DIR_FASE1_SEM_VENTO = './resultados/'
DIR_FASE1_COM_VENTO = './resultados2/'
DIR_FASE2           = './resultados_p2/'
DIR_GRAFICOS        = './graficos/'

CSV_FASE1_SEM_VENTO = os.path.join(DIR_FASE1_SEM_VENTO, 'test_results.csv')
CSV_FASE1_COM_VENTO = os.path.join(DIR_FASE1_COM_VENTO, 'test_results.csv')
CSV_FASE2           = os.path.join(DIR_FASE2,           'test_results_p2.csv')

N_RUNS = 5


def run(show=False):
    os.makedirs(DIR_GRAFICOS, exist_ok=True)

    data_sem   = None
    data_com   = None

    # ----------------------------------------------------------------
    # Fase 1 — Sem Vento
    # ----------------------------------------------------------------
    if os.path.isdir(DIR_FASE1_SEM_VENTO):
        print('\n' + '='*60)
        print('Fase 1 — Sem Vento')
        print('='*60)
        data_sem = pr.load_all(DIR_FASE1_SEM_VENTO, N_RUNS)
        test_sem = pr.load_test_results(CSV_FASE1_SEM_VENTO)

        pr.print_terminal_table(data_sem, test_sem, label='Sem Vento')
        pr.fig1_all_experiments(data_sem, DIR_GRAFICOS, show, label='sem_vento')
        pr.fig2_comparison     (data_sem, DIR_GRAFICOS, show, label='sem_vento')
        pr.fig3_factors        (data_sem, DIR_GRAFICOS, show, label='sem_vento')
        pr.fig4_success_bar    (test_sem, DIR_GRAFICOS, show, label='sem_vento')
        pr.fig6_summary_table  (data_sem, test_sem, DIR_GRAFICOS, show, label='sem_vento')
    else:
        print(f'[aviso] {DIR_FASE1_SEM_VENTO!r} não encontrada — a saltar Fase 1 sem vento.')

    # ----------------------------------------------------------------
    # Fase 1 — Com Vento
    # ----------------------------------------------------------------
    if os.path.isdir(DIR_FASE1_COM_VENTO):
        print('\n' + '='*60)
        print('Fase 1 — Com Vento')
        print('='*60)
        data_com = pr.load_all(DIR_FASE1_COM_VENTO, N_RUNS)
        test_com = pr.load_test_results(CSV_FASE1_COM_VENTO)

        pr.print_terminal_table(data_com, test_com, label='Com Vento')
        pr.fig1_all_experiments(data_com, DIR_GRAFICOS, show, label='com_vento')
        pr.fig2_comparison     (data_com, DIR_GRAFICOS, show, label='com_vento')
        pr.fig3_factors        (data_com, DIR_GRAFICOS, show, label='com_vento')
        pr.fig4_success_bar    (test_com, DIR_GRAFICOS, show, label='com_vento')
        pr.fig6_summary_table  (data_com, test_com, DIR_GRAFICOS, show, label='com_vento')
    else:
        print(f'[aviso] {DIR_FASE1_COM_VENTO!r} não encontrada — a saltar Fase 1 com vento.')

    # ----------------------------------------------------------------
    # Comparação Sem vs Com Vento
    # ----------------------------------------------------------------
    if data_sem is not None and data_com is not None:
        print('\n' + '='*60)
        print('Comparação Sem Vento vs Com Vento')
        print('='*60)
        pr.fig5_wind_comparison(data_sem, data_com, DIR_GRAFICOS, show)

    # ----------------------------------------------------------------
    # Fase 2 — Comparação de Operadores
    # ----------------------------------------------------------------
    if os.path.isdir(DIR_FASE2):
        print('\n' + '='*60)
        print('Fase 2 — Comparação de Operadores Genéticos')
        print('='*60)
        p2_data = pr.load_p2_all(DIR_FASE2, N_RUNS)
        test_p2 = pr.load_p2_test_results(CSV_FASE2)

        has_any = any(v is not None for v in p2_data.values())
        if has_any:
            pr.fig_p2_curves(p2_data, DIR_GRAFICOS, show)
            pr.fig_p2_bars  (p2_data, test_p2, DIR_GRAFICOS, show)
        else:
            print(f'[aviso] Nenhum log encontrado em {DIR_FASE2!r}.')
    else:
        print(f'[aviso] {DIR_FASE2!r} não encontrada — a saltar Fase 2.')

    # ----------------------------------------------------------------
    # Sumário final
    # ----------------------------------------------------------------
    print('\n' + '='*60)
    print('Concluído.')
    print(f'Gráficos guardados em: {os.path.abspath(DIR_GRAFICOS)}')
    pdfs = sorted(f for f in os.listdir(DIR_GRAFICOS) if f.endswith('.pdf'))
    if pdfs:
        print()
        for name in pdfs:
            print(f'  {name}')
    print('='*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera todos os gráficos do TP2.')
    parser.add_argument('--show', action='store_true',
                        help='Abre janelas interativas em vez de guardar PDF')
    args = parser.parse_args()
    run(show=args.show)
