import random
import copy
import numpy as np
import gymnasium as gym
import os
from multiprocessing import Process, Queue

# ============================================================
# CONFIG — alterar estes valores para mudar o comportamento
# ============================================================
ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 100     # episódios para avaliação final (1000 para relatório)
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()

nInputs = 8
nOutputs = 2
SHAPE = (nInputs, 12, nOutputs)

GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i - 1] * SHAPE[i]   # 8*12 + 12*2 = 120

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100

# Estes três parâmetros são sobrepostos pelo ciclo de experiências em __main__
PROB_CROSSOVER = 0.9
PROB_MUTATION  = 0.008   # ≈ 1/GENOTYPE_SIZE
STD_DEV        = 0.1
ELITE_SIZE     = 1


# ============================================================
# Rede neuronal
# ============================================================
def network(shape, observation, ind):
    """Calcula a acção da rede neuronal dado o genótipo e a observação."""
    x = observation[:]
    for i in range(1, len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k] * ind[k + j * len(x)]
        x = np.tanh(y)
    return x


# ============================================================
# Simulação e função objectivo
# ============================================================
def check_successful_landing(observation):
    """Verifica se a aterragem foi bem-sucedida segundo os critérios do enunciado."""
    x              = observation[0]
    vy             = observation[3]
    theta          = observation[4]
    contact_left   = observation[6]
    contact_right  = observation[7]

    legs_touching      = contact_left == 1 and contact_right == 1
    on_landing_pad     = abs(x) <= 0.2
    stable_velocity    = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)

    return legs_touching and on_landing_pad and stable_velocity and stable_orientation


def objective_function(observation):
    """
    Função objectivo com gradiente suave — avalia a última observação antes do impacto.
    Penaliza distância ao pad, velocidade e ângulo de forma contínua.
    Bónus de estabilidade quando ambas as pernas tocam. Sucesso vale +50 (sem cliff).
    Retorna (fitness, sucesso_bool).
    """
    x, y, vx, vy, theta, v_theta, contact_left, contact_right = observation

    distance_to_pad    = np.sqrt(x**2 + y**2)
    velocity_magnitude = np.sqrt(vx**2 + vy**2)

    distance_reward = -3.0 * distance_to_pad
    velocity_penalty = -2.0 * velocity_magnitude
    angle_penalty    = -1.0 * (abs(theta) + 0.5 * abs(v_theta))

    leg_bonus = (10 if contact_left == 1 else 0) + (10 if contact_right == 1 else 0)

    landing_stability_bonus = 0
    if contact_left == 1 and contact_right == 1:
        landing_stability_bonus += 20 * np.exp(-velocity_magnitude)
        landing_stability_bonus += 20 * np.exp(-abs(theta))
        landing_stability_bonus += 10 * np.exp(-abs(x) / 0.2)

    fitness = distance_reward + velocity_penalty + angle_penalty + leg_bonus + landing_stability_bonus

    if check_successful_landing(observation):
        fitness += 50

    return fitness, check_successful_landing(observation)


def simulate(genotype, render_mode=None, seed=None, env=None):
    """Simula um episódio do Lunar Lander e avalia o indivíduo."""
    env_was_none = env is None
    if env is None:
        env = gym.make(
            "LunarLander-v3", render_mode=render_mode,
            continuous=True, gravity=GRAVITY,
            enable_wind=ENABLE_WIND, wind_power=WIND_POWER,
            turbulence_power=TURBULENCE_POWER
        )

    observation, info = env.reset(seed=seed)

    for _ in range(STEPS):
        prev_observation = observation
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    if env_was_none:
        env.close()

    # Avalia a última observação antes do impacto/paragem
    return objective_function(prev_observation)


# ============================================================
# Avaliação paralela
# ============================================================
def evaluate(evaluationQueue, evaluatedQueue):
    """Worker: avalia indivíduos até receber None."""
    env = gym.make(
        "LunarLander-v3", render_mode=None,
        continuous=True, gravity=GRAVITY,
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER
    )
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind['fitness'] = simulate(ind['genotype'], seed=None, env=env)[0]
        evaluatedQueue.put(ind)
    env.close()


def evaluate_population(population):
    """Avalia uma lista de indivíduos usando múltiplos processos."""
    for ind in population:
        evaluationQueue.put(ind)
    new_pop = []
    for _ in range(len(population)):
        new_pop.append(evaluatedQueue.get())
    return new_pop


# ============================================================
# Operadores genéticos
# ============================================================
def generate_initial_population():
    """Gera a população inicial com genótipos uniformes em [-1, 1]."""
    population = []
    for _ in range(POPULATION_SIZE):
        genotype = [random.uniform(-1, 1) for _ in range(GENOTYPE_SIZE)]
        population.append({'genotype': genotype, 'fitness': None})
    return population


def parent_selection(population):
    """Selecção por torneio (tamanho 3) — selecciona o melhor dos 3 sorteados."""
    tournament = random.sample(population, 3)
    winner = max(tournament, key=lambda ind: ind['fitness'])
    return copy.deepcopy(winner)


def crossover(p1, p2):
    """Crossover uniforme: cada gene herdado de p1 ou p2 com probabilidade 0.5."""
    genotype = [
        p1['genotype'][i] if random.random() < 0.5 else p2['genotype'][i]
        for i in range(len(p1['genotype']))
    ]
    return {'genotype': genotype, 'fitness': None}


def mutation(p):
    """Mutação gaussiana: cada gene perturbado com probabilidade PROB_MUTATION."""
    ind = copy.deepcopy(p)
    for i in range(len(ind['genotype'])):
        if random.random() < PROB_MUTATION:
            ind['genotype'][i] += random.gauss(0, STD_DEV)
    ind['fitness'] = None
    return ind


def survival_selection(population, offspring):
    """
    Selecção de sobreviventes com elitismo.
    Os ELITE_SIZE melhores da população anterior são reavaliados e preservados.
    Quando ELITE_SIZE=0 equivale a substituição completa pela descendência.
    """
    offspring.sort(key=lambda x: x['fitness'], reverse=True)
    elite = evaluate_population(population[:ELITE_SIZE])
    new_population = elite + offspring[ELITE_SIZE:]
    new_population.sort(key=lambda x: x['fitness'], reverse=True)
    return new_population


# ============================================================
# Ciclo evolutivo
# ============================================================
def evolution():
    """
    Executa o algoritmo evolucionário.
    Usa os parâmetros globais: PROB_MUTATION, PROB_CROSSOVER, ELITE_SIZE,
    POPULATION_SIZE, NUMBER_OF_GENERATIONS.
    Retorna lista de (genótipo, fitness) do melhor por geração.
    """
    evaluation_processes = []
    for _ in range(NUM_PROCESSES):
        p = Process(target=evaluate, args=(evaluationQueue, evaluatedQueue))
        p.start()
        evaluation_processes.append(p)

    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key=lambda x: x['fitness'], reverse=True)

    bests = [(population[0]['genotype'], population[0]['fitness'])]

    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)
            else:
                ni = parent_selection(population)
            ni = mutation(ni)
            offspring.append(ni)

        offspring = evaluate_population(offspring)
        population = survival_selection(population, offspring)

        best = (population[0]['genotype'], population[0]['fitness'])
        bests.append(best)
        print(f'  Gen {gen:3d}: {best[1]:.2f}')

    for _ in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()

    return bests


# ============================================================
# Carregar logs
# ============================================================
def load_bests(fname):
    """Carrega o ficheiro de log e retorna lista de (fitness, shape, genótipo)."""
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append((eval(fitness), eval(shape), eval(genotype)))
    return bests


# ============================================================
# Ponto de entrada
# ============================================================
if __name__ == '__main__':

    # ---- ESCOLHER MODO ----
    # 'evolve'   : correr as 8 experiências (Tabela 2)
    # 'test'     : testar um indivíduo específico (sem visualização)
    # 'test_all' : testa o melhor indivíduo de cada log e gera test_results.csv
    # 'view'     : visualizar indivíduo evoluído com janela
    MODE = 'evolve'

    # ================================================================
    # MODO EVOLVE — executa as 8 experiências da Tabela 2
    # ================================================================
    if MODE == 'evolve':

        # Tabela 2 do enunciado: (mutação, crossover, elitismo)
        EXPERIMENTS = [
            {'mut': 0.008, 'cx': 0.5, 'elite': 0},  # Experiência 1
            {'mut': 0.05,  'cx': 0.5, 'elite': 0},  # Experiência 2
            {'mut': 0.008, 'cx': 0.9, 'elite': 0},  # Experiência 3
            {'mut': 0.05,  'cx': 0.9, 'elite': 0},  # Experiência 4
            {'mut': 0.008, 'cx': 0.5, 'elite': 1},  # Experiência 5
            {'mut': 0.05,  'cx': 0.5, 'elite': 1},  # Experiência 6
            {'mut': 0.008, 'cx': 0.9, 'elite': 1},  # Experiência 7
            {'mut': 0.05,  'cx': 0.9, 'elite': 1},  # Experiência 8
        ]

        N_RUNS = 5
        # 40 seeds (8 experiências × 5 runs) para reprodutibilidade
        SEEDS = [
            964, 952, 364, 913, 140,   # Exp 1
            726, 112, 631, 881, 844,   # Exp 2
            965, 672, 335, 611, 457,   # Exp 3
            591, 551, 538, 673, 437,   # Exp 4
            513, 893, 709, 489, 788,   # Exp 5
            709, 751, 467, 596, 976,   # Exp 6
            101, 202, 303, 404, 505,   # Exp 7
            606, 707, 808, 909, 110,   # Exp 8
        ]

        # Para correr apenas experiências específicas, alterar este range
        # Ex: range(0, 1) corre só a Experiência 1
        EXP_RANGE = range(len(EXPERIMENTS))

        for exp_idx in EXP_RANGE:
            config  = EXPERIMENTS[exp_idx]
            exp_num = exp_idx + 1

            # Atualiza os parâmetros globais usados pelos operadores
            PROB_MUTATION  = config['mut']
            PROB_CROSSOVER = config['cx']
            ELITE_SIZE     = config['elite']

            print(f'\n{"="*60}')
            print(f'Experiência {exp_num}: mut={PROB_MUTATION}, '
                  f'cx={PROB_CROSSOVER}, elite={ELITE_SIZE}')
            print(f'{"="*60}')

            for run in range(N_RUNS):
                seed = SEEDS[exp_idx * N_RUNS + run]
                random.seed(seed)
                np.random.seed(seed)

                print(f'\n  --- Run {run + 1}/{N_RUNS} (seed={seed}) ---')
                bests = evolution()

                fname = f'log_e{exp_num}_r{run}.txt'
                with open(fname, 'w') as f:
                    for b in bests:
                        f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                print(f'  Saved {fname}  |  best fitness = {bests[-1][1]:.2f}')

    # ================================================================
    # MODO TEST — avalia indivíduo evoluído sem visualização
    # ================================================================
    elif MODE == 'test':

        # Alterar para o ficheiro desejado
        filename  = 'log_e1_r0.txt'
        n_test_ep = TEST_EPISODES   # 100 rápido; 1000 para relatório

        bests = load_bests(filename)
        b     = bests[-1]   # última geração = melhor encontrado
        SHAPE = b[1]
        ind   = b[2]

        total_fit, total_success = 0.0, 0
        for i in range(n_test_ep):
            f, s = simulate(ind, render_mode=None, seed=None)
            total_fit     += f
            total_success += int(s)

        print(f'\n{"="*40}')
        print(f'Ficheiro     : {filename}')
        print(f'Episódios    : {n_test_ep}')
        print(f'Fitness médio: {total_fit / n_test_ep:.4f}')
        print(f'Taxa sucesso : {total_success / n_test_ep:.4f} '
              f'({total_success}/{n_test_ep})')
        print(f'{"="*40}')

    # ================================================================
    # MODO TEST_ALL — testa todos os logs e gera test_results.csv
    # ================================================================
    elif MODE == 'test_all':
        import csv

        # Pasta onde estão os logs (relativa ao local de execução)
        LOG_DIR   = '../resultados/'   # alterar para '../resultados2/' para com vento
        N_TEST_EP = TEST_EPISODES      # 100 rápido; usar 1000 para relatório final
        OUT_CSV   = os.path.join(LOG_DIR, 'test_results.csv')

        EXP_CONFIGS = [
            {'mut': 0.008, 'cx': 0.5, 'elite': 0},
            {'mut': 0.05,  'cx': 0.5, 'elite': 0},
            {'mut': 0.008, 'cx': 0.9, 'elite': 0},
            {'mut': 0.05,  'cx': 0.9, 'elite': 0},
            {'mut': 0.008, 'cx': 0.5, 'elite': 1},
            {'mut': 0.05,  'cx': 0.5, 'elite': 1},
            {'mut': 0.008, 'cx': 0.9, 'elite': 1},
            {'mut': 0.05,  'cx': 0.9, 'elite': 1},
        ]

        rows = []
        for exp_idx, config in enumerate(EXP_CONFIGS):
            exp_num = exp_idx + 1
            print(f'\nExperiência {exp_num}: mut={config["mut"]}, '
                  f'cx={config["cx"]}, elite={config["elite"]}')

            for run in range(5):
                fname = os.path.join(LOG_DIR, f'log_e{exp_num}_r{run}.txt')
                if not os.path.exists(fname):
                    print(f'  [aviso] {fname} não encontrado — a saltar.')
                    continue

                bests = load_bests(fname)
                b = bests[-1]
                SHAPE = b[1]
                ind   = b[2]

                total_fit, total_success = 0.0, 0
                for _ in range(N_TEST_EP):
                    f, s = simulate(ind, render_mode=None, seed=None)
                    total_fit     += f
                    total_success += int(s)

                mean_fit = total_fit / N_TEST_EP
                succ_rate = total_success / N_TEST_EP
                print(f'  Run {run}: fitness={mean_fit:.2f}, sucesso={succ_rate:.3f}')
                rows.append({
                    'exp': exp_num, 'run': run,
                    'mean_fitness': round(mean_fit, 4),
                    'success_rate': round(succ_rate, 4),
                })

        with open(OUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['exp', 'run', 'mean_fitness', 'success_rate'])
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nResultados guardados em: {OUT_CSV}')

    # ================================================================
    # MODO VIEW — visualiza o indivíduo numa janela
    # ================================================================
    elif MODE == 'view':

        filename = 'log_e1_r0.txt'

        bests = load_bests(filename)
        b     = bests[-1]
        SHAPE = b[1]
        ind   = b[2]

        print(f'A visualizar {filename} (fitness={b[0]:.2f}) ...')
        simulate(ind, render_mode='human', seed=None)
