"""
run_pipeline.py — Corre todo o pipeline do TP2 de uma vez.

Uso:
    python3 run_pipeline.py                  # Fase 1 + Fase 2 + gráficos
    python3 run_pipeline.py --skip-evolve    # só gráficos (logs já existem)
    python3 run_pipeline.py --only-phase2    # só Fase 2 + gráficos

Ordem de execução:
    1. Fase 1  → evolve      → ./resultados/      (log_e*_r*.txt)
    2. Fase 2  → evolve_ph2  → ./resultados_p2/   (log_p2_*_r*.txt)
    3. Gráficos → gerar_graficos.py → ./graficos/
"""

import os
import sys
import subprocess
import argparse
import time

HERE       = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(HERE, 'NE-LunarLander-alunos (1).py')
PLOT_SCRIPT = os.path.join(HERE, 'gerar_graficos.py')
PYTHON      = sys.executable   # usa o mesmo python3 que está a correr este script


def run_step(label, cmd):
    """Corre um passo e imprime cabeçalho/rodapé com tempo."""
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'\n[ERRO] "{label}" terminou com código {result.returncode}.')
        sys.exit(result.returncode)
    print(f'\n  Concluído em {elapsed/60:.1f} min.')


def main():
    parser = argparse.ArgumentParser(description='Pipeline completo do TP2.')
    parser.add_argument('--skip-evolve', action='store_true',
                        help='Salta a evolução (usa logs já existentes)')
    parser.add_argument('--only-phase2', action='store_true',
                        help='Corre só a Fase 2 e os gráficos')
    parser.add_argument('--skip-graphs', action='store_true',
                        help='Salta a geração de gráficos')
    args = parser.parse_args()

    t_total = time.time()

    if not args.skip_evolve:
        if not args.only_phase2:
            run_step(
                'FASE 1 — 8 experiências (Crossover Uniforme + Mut Gaussiana)',
                [PYTHON, MAIN_SCRIPT, 'evolve'],
            )
        run_step(
            'FASE 2 — 6 combinações de operadores (Com Vento)',
            [PYTHON, MAIN_SCRIPT, 'evolve_phase2'],
        )
    else:
        print('\n[--skip-evolve] A saltar evolução — a usar logs existentes.')

    if not args.skip_graphs:
        run_step(
            'GRÁFICOS — a gerar todos os PDFs',
            [PYTHON, PLOT_SCRIPT],
        )

    elapsed_total = time.time() - t_total
    print(f'\n{"="*60}')
    print(f'  Pipeline concluído em {elapsed_total/60:.1f} min.')
    print(f'  Logs:     ./resultados/  e  ./resultados_p2/')
    print(f'  Gráficos: ./graficos/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
