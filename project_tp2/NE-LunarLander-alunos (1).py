import random
import copy
import numpy as np
import gymnasium as gym
import os
from multiprocessing import Process, Queue

# ============================================================
# CONFIG — parâmetros físicos e estruturais (não alterar)
# ============================================================
ENABLE_WIND      = True
WIND_POWER       = 15.0
TURBULENCE_POWER = 0.0
GRAVITY          = -10.0
RENDER_MODE      = 'human'
STEPS            = 500

NUM_PROCESSES   = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue  = Queue()

nInputs  = 8
nOutputs = 2
SHAPE    = (nInputs, 12, nOutputs)

GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i - 1] * SHAPE[i]   # 8*12 + 12*2 = 120

POPULATION_SIZE = 100

# Hiperparâmetros de treino — sobrepostos em __main__
NUMBER_OF_GENERATIONS = 150
PROB_CROSSOVER        = 0.9
PROB_MUTATION         = 0.05
STD_DEV               = 0.1
ELITE_SIZE            = 2
TOURNAMENT_SIZE       = 5


# ============================================================
# REDE NEURONAL
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
# FUNÇÃO OBJECTIVO
# ============================================================
def check_successful_landing(observation):
    """Verifica se a aterragem foi bem-sucedida segundo os critérios do enunciado."""
    x            = observation[0]
    vy           = observation[3]
    theta        = observation[4]
    contact_left = observation[6]
    contact_right= observation[7]

    legs_touching      = contact_left == 1 and contact_right == 1
    on_landing_pad     = abs(x) <= 0.2
    stable_velocity    = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)

    return legs_touching and on_landing_pad and stable_velocity and stable_orientation


def objective_function(observation):
    """
    Fase 1/2 — gradiente suave.
    Penaliza distância ao pad, velocidade e ângulo de forma contínua.
    Bónus de estabilidade quando ambas as pernas tocam. Sucesso vale +50.
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


def objective_function_v2(observation):
    """
    Fase 3 — reward shaping extremo.
    Penalidades quadráticas no ângulo e velocidade.
    Bónus exponenciais massivos para aterragem suave com as duas pernas.
    Sucesso vale +100. Retorna (fitness, sucesso_bool).
    """
    x, y, vx, vy, theta, v_theta, contact_left, contact_right = observation

    distance_to_pad    = np.sqrt(x**2 + y**2)
    velocity_magnitude = np.sqrt(vx**2 + vy**2)

    distance_reward  = -5.0 * distance_to_pad
    velocity_penalty = -3.0 * (vx**2 + vy**2)
    angle_penalty    = -5.0 * theta**2 - 2.0 * v_theta**2

    leg_bonus = (10 if contact_left == 1 else 0) + (10 if contact_right == 1 else 0)

    landing_bonus = 0
    if contact_left == 1 and contact_right == 1:
        landing_bonus += 100 * np.exp(-3.0 * velocity_magnitude)
        landing_bonus += 100 * np.exp(-5.0 * abs(theta))
        landing_bonus +=  50 * np.exp(-abs(x) / 0.15)

    fitness = distance_reward + velocity_penalty + angle_penalty + leg_bonus + landing_bonus

    if check_successful_landing(observation):
        fitness += 100

    return fitness, check_successful_landing(observation)


# ============================================================
# SIMULAÇÃO
# ============================================================
def simulate(genotype, render_mode=None, seed=None, env=None):
    """Fase 1/2 — simula um episódio e avalia com objective_function."""
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

    return objective_function(prev_observation)


def simulate_v2(genotype, render_mode=None, seed=None, env=None):
    """Fase 3 — simula um episódio e avalia com objective_function_v2."""
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

    return objective_function_v2(prev_observation)


# ============================================================
# AVALIAÇÃO PARALELA
# ============================================================
def evaluate(evaluationQueue, evaluatedQueue):
    """Worker Fase 1/2: usa simulate (objective_function)."""
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


def evaluate_v2(evaluationQueue, evaluatedQueue):
    """Worker Fase 3: usa simulate_v2 (objective_function_v2)."""
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
        ind['fitness'] = simulate_v2(ind['genotype'], seed=None, env=env)[0]
        evaluatedQueue.put(ind)
    env.close()


def evaluate_population(population):
    """Avalia uma lista de indivíduos usando os workers activos."""
    for ind in population:
        evaluationQueue.put(ind)
    new_pop = []
    for _ in range(len(population)):
        new_pop.append(evaluatedQueue.get())
    return new_pop


# ============================================================
# GESTÃO DA POPULAÇÃO
# ============================================================
def generate_initial_population():
    """Gera a população inicial com genótipos uniformes em [-1, 1]."""
    population = []
    for _ in range(POPULATION_SIZE):
        genotype = [random.uniform(-1, 1) for _ in range(GENOTYPE_SIZE)]
        population.append({'genotype': genotype, 'fitness': None})
    return population


def survival_selection(population, offspring):
    """
    Selecção de sobreviventes com elitismo.
    Os ELITE_SIZE melhores da população anterior são reavaliados e preservados.
    ELITE_SIZE=0 equivale a substituição completa pela descendência.
    """
    offspring.sort(key=lambda x: x['fitness'], reverse=True)
    elite = evaluate_population(population[:ELITE_SIZE])
    new_population = elite + offspring[ELITE_SIZE:]
    new_population.sort(key=lambda x: x['fitness'], reverse=True)
    return new_population


# ============================================================
# SELECÇÃO DE PAIS
# ============================================================
def parent_selection(population):
    """Selecção por torneio — selecciona o melhor dos TOURNAMENT_SIZE sorteados."""
    k = globals().get('TOURNAMENT_SIZE', 3)
    tournament = random.sample(population, k)
    winner = max(tournament, key=lambda ind: ind['fitness'])
    return copy.deepcopy(winner)


# ============================================================
# CROSSOVER
# ============================================================
def crossover(p1, p2):
    """Fase 1 — crossover uniforme: cada gene herdado de p1 ou p2 com prob. 0.5."""
    genotype = [
        p1['genotype'][i] if random.random() < 0.5 else p2['genotype'][i]
        for i in range(len(p1['genotype']))
    ]
    return {'genotype': genotype, 'fitness': None}


def crossover_two_point(p1, p2):
    """Fase 2 — crossover de dois pontos: p1 | p2 | p1."""
    size = len(p1['genotype'])
    c1, c2 = sorted(random.sample(range(size + 1), 2))
    genotype = (
        p1['genotype'][:c1] +
        p2['genotype'][c1:c2] +
        p1['genotype'][c2:]
    )
    return {'genotype': genotype, 'fitness': None}


def crossover_arithmetic(p1, p2):
    """Fase 2 — crossover aritmético: filho[i] = alfa*p1[i] + (1-alfa)*p2[i]."""
    alfa = random.random()
    genotype = [
        alfa * p1['genotype'][i] + (1 - alfa) * p2['genotype'][i]
        for i in range(len(p1['genotype']))
    ]
    return {'genotype': genotype, 'fitness': None}


def crossover_adaptive(p1, p2):
    """
    Fase 3 — crossover adaptativo baseado no fitness dos pais.
    - Ambos >= 119 : BLX-alpha (alpha=0.01), explotação fina.
    - Um >= 100, outro < 100 : aritmético ponderado 90%/10%.
    - Caso contrário : uniforme 50/50 para exploração.
    offspring['fitness'] = None garante reavaliação no simulador.
    """
    f1 = p1['fitness'] if p1['fitness'] is not None else 0
    f2 = p2['fitness'] if p2['fitness'] is not None else 0
    size = len(p1['genotype'])

    if f1 >= 119 and f2 >= 119:
        alpha = 0.01
        genotype = []
        for i in range(size):
            g1, g2 = p1['genotype'][i], p2['genotype'][i]
            lo = min(g1, g2)
            hi = max(g1, g2)
            spread = (hi - lo) * alpha
            genotype.append(random.uniform(lo - spread, hi + spread))

    elif (f1 >= 100) != (f2 >= 100):
        good = p1 if f1 >= 100 else p2
        bad  = p2 if f1 >= 100 else p1
        genotype = [
            0.9 * good['genotype'][i] + 0.1 * bad['genotype'][i]
            for i in range(size)
        ]

    else:
        genotype = [
            p1['genotype'][i] if random.random() < 0.5 else p2['genotype'][i]
            for i in range(size)
        ]

    offspring = {'genotype': genotype, 'fitness': None}
    return offspring


# ============================================================
# MUTAÇÃO
# ============================================================
def mutation(p):
    """Fase 1 — mutação gaussiana: perturbação += gauss(0, STD_DEV) por gene."""
    ind = copy.deepcopy(p)
    for i in range(len(ind['genotype'])):
        if random.random() < PROB_MUTATION:
            ind['genotype'][i] += random.gauss(0, STD_DEV)
    ind['fitness'] = None
    return ind


def mutation_uniform(p):
    """Fase 2 — mutação uniforme: perturbação += uniform(-0.2, 0.2) por gene."""
    ind = copy.deepcopy(p)
    for i in range(len(ind['genotype'])):
        if random.random() < PROB_MUTATION:
            ind['genotype'][i] += random.uniform(-0.2, 0.2)
    ind['fitness'] = None
    return ind


def mutation_adaptive(p):
    """
    Fase 3 — mutação com intensidade adaptada ao fitness do indivíduo.
    - fitness None ou < 100 : perturbação forte  (+= uniform(-1.5, 1.5))
    - fitness em [100, 118[ : perturbação média  (+= gauss(0, 0.3))
    - fitness >= 118        : micro-ajustes finos (+= gauss(0, 0.05))
    Usa sempre += para preservar a herança genética do crossover.
    """
    ind = copy.deepcopy(p)
    fitness = ind.get('fitness', None)
    for i in range(len(ind['genotype'])):
        if random.random() < PROB_MUTATION:
            if fitness is None or fitness < 100:
                ind['genotype'][i] += random.uniform(-1.5, 1.5)
            elif fitness < 118:
                ind['genotype'][i] += random.gauss(0, 0.3)
            else:
                ind['genotype'][i] += random.gauss(0, 0.05)
    ind['fitness'] = None
    return ind


# ============================================================
# CICLO EVOLUTIVO
# ============================================================
def evolution():
    """
    Fase 1 — algoritmo base com crossover uniforme e mutação gaussiana.
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


def evolution_p2(cx_fn, mut_fn):
    """
    Fase 2 — aceita operadores de crossover e mutação como argumentos.
    Usa os parâmetros globais sem os alterar.
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
                ni = cx_fn(p1, p2)
            else:
                ni = parent_selection(population)
            ni = mut_fn(ni)
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


def evolution_p3(cx_fn, mut_fn):
    """
    Fase 3 — idêntico a evolution_p2 mas os workers usam simulate_v2
    (evaluate_v2), aplicando objective_function_v2 em cada avaliação.
    Retorna lista de (genótipo, fitness) do melhor por geração.
    """
    evaluation_processes = []
    for _ in range(NUM_PROCESSES):
        p = Process(target=evaluate_v2, args=(evaluationQueue, evaluatedQueue))
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
                ni = cx_fn(p1, p2)
            else:
                ni = parent_selection(population)
            ni = mut_fn(ni)
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
# UTILITÁRIOS
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
# PONTO DE ENTRADA — Produto Final
# ============================================================
if __name__ == '__main__':

    # --- Passo A: Hiperparâmetros finais optimizados ---
    ENABLE_WIND           = True
    PROB_MUTATION         = 0.05
    PROB_CROSSOVER        = 0.9
    ELITE_SIZE            = 2
    TOURNAMENT_SIZE       = 5
    NUMBER_OF_GENERATIONS = 150

    # --- Passo B: Treino com os melhores operadores (Fase 3) ---
    print('A iniciar o treino do modelo otimizado (Fase 3)...')
    print(f'  Config: mut={PROB_MUTATION}, cx={PROB_CROSSOVER}, '
          f'elite={ELITE_SIZE}, torneio={TOURNAMENT_SIZE}, '
          f'gerações={NUMBER_OF_GENERATIONS}')
    bests = evolution_p3(crossover_adaptive, mutation_adaptive)

    # --- Passo C: Extracção do melhor genótipo ---
    best_genotype = bests[-1][0]
    best_fitness  = bests[-1][1]
    print(f'\nTreino concluído! Melhor fitness de treino: {best_fitness:.4f}')
    print('A avaliar o melhor agente em 1000 episódios...')

    # --- Passo D: Avaliação final e apresentação dos resultados ---
    N_TEST = 1000
    total_fit, total_success = 0.0, 0
    for _ in range(N_TEST):
        f, s = simulate(best_genotype, render_mode=None, seed=None)
        total_fit     += f
        total_success += int(s)

    print(f'\n{"="*50}')
    print(f'Fitness Médio  : {total_fit / N_TEST:.4f}')
    print(f'Taxa de Sucesso: {total_success / N_TEST * 100:.1f}% ({total_success}/{N_TEST})')
    print(f'{"="*50}')
