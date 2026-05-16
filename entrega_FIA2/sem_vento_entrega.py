import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = 0.9 #1 #ocorre sempre

  
PROB_MUTATION = 0.05 #1.0/GENOTYPE_SIZE
STD_DEV = 0.05


ELITE_SIZE = 1 #1 #posso aumentar isto



def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False


def network(shape, observation, genotype): #a original estava a dar bosta devido á arquitetura escolhida
    x = observation[:]  # Cópia dos inputs
    offset = 0          # Inicializa o offset para o bloco de pesos atual
    for i in range(1, len(shape)):
        input_size = shape[i - 1]
        output_size = shape[i]
        y = np.zeros(output_size)
        # Para cada neurônio da camada i (output layer)
        for j in range(output_size):
            # Para cada entrada da camada atual
            for k in range(input_size):
                y[j] += x[k] * genotype[offset + k + j * input_size]
        x = np.tanh(y)
        # Atualiza o offset para passar para os pesos da próxima camada
        offset += input_size * output_size
    return x

def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    for _ in range(STEPS):
        prev_observation = observation
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(prev_observation)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population






def parent_selection1(population):
    #Seleção de pais com base numa populacao de 5 genotipos escolhidos ao acaso
    tournament_size = 5
    participants = random.sample(population, tournament_size)
    return copy.deepcopy(max(participants, key=lambda ind: ind['fitness']))

def parent_selection2(population, tournament_size=10, selection_pressure=0.75): 
        
    # Seleciona candidatos aleatoriamente para o torneio
    candidates = random.sample(population, tournament_size)
    
    # Ordena os candidatos da melhor para a pior fitness
    candidates.sort(key=lambda ind: ind['fitness'], reverse=True)
    
    # Percorre os candidatos: com probabilidade selection_pressure, escolhe o candidato atual
    for candidate in candidates:
        if random.random() < selection_pressure:
            return copy.deepcopy(candidate)
    
    # Se nenhum candidato for escolhido pela probabilidade, retorna o menos apto (ultimo da lista)
    return copy.deepcopy(candidates[-1])


def parent_selection(population, tournament_size=7, base_selection_pressure=0.75, fitness_threshold=115):
    candidates = random.sample(population, tournament_size)
    
    # Ordena os candidatos do melhor para o pior fitness
    candidates.sort(key=lambda ind: ind['fitness'], reverse=True)
    
    if candidates[0]['fitness'] >= fitness_threshold:
        selection_pressure = min(1.0, base_selection_pressure + 0.15)  # Por exemplo, sobe para 0.90
    else:
        selection_pressure = base_selection_pressure

    
    for candidate in candidates:
        if random.random() < selection_pressure:
            return copy.deepcopy(candidate)
    
    return copy.deepcopy(candidates[-1])






def crossover1(p1, p2):
    offspring = copy.deepcopy(p1)
    if random.random() < PROB_CROSSOVER:
        crossover_point = random.randint(1, len(p1['genotype']) - 1)
        offspring['genotype'] = p1['genotype'][:crossover_point] + p2['genotype'][crossover_point:]
    return offspring


def crossover2(p1, p2):

    offspring = copy.deepcopy(p1)
    if random.random() < PROB_CROSSOVER:
        f1 = p1.get('fitness', 0)
        f2 = p2.get('fitness', 0)
        
        # Verifica se há um pai com fitness muito superior: se sim faz se um crossover aritmético ponderado (90% do pai com fitness alto e 10% do outro)
        if (f1 >= 100 and f2 < 100) or (f2 >= 100 and f1 < 100):
            # Crossover aritmético ponderado
            if f1 >= 100 and f2 < 100:
                weight1, weight2 = 0.9, 0.1
            else:
                weight1, weight2 = 0.1, 0.9
            new_genotype = []
            for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
                new_gene = weight1 * gene1 + weight2 * gene2
                new_genotype.append(new_gene)
            offspring['genotype'] = new_genotype
        else:
            # Crossover uniforme: para cada gene, escolhe aleatoriamente de p1 ou p2 aumenta o espaço procurado, ajuda a ser mais rapido a encontrar a solução
            new_genotype = []
            for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
                if random.random() < 0.5:
                    new_genotype.append(gene1)
                else:
                    new_genotype.append(gene2)
            offspring['genotype'] = new_genotype
    return offspring


def crossover(p1, p2):
    offspring = copy.deepcopy(p1)
    if random.random() < PROB_CROSSOVER:
        f1 = p1.get('fitness', 0)
        f2 = p2.get('fitness', 0)

        if f1 >= 119 and f2 >= 119:
            # BLX-alpha crossover para precisão quando temos 2 pais bons
            alpha = 0.01  
            new_genotype = []
            for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
                min_gene = min(gene1, gene2)
                max_gene = max(gene1, gene2)
                range_gene = max_gene - min_gene
                lower_bound = min_gene - alpha * range_gene
                upper_bound = max_gene + alpha * range_gene
                new_gene = random.uniform(lower_bound, upper_bound)
                new_genotype.append(new_gene)
            offspring['genotype'] = new_genotype

        elif (f1 >= 100 and f2 < 100) or (f2 >= 100 and f1 < 100):
            # Crossover aritmético ponderado que favorece o pai que é bastante melhor que o outro
            if f1 >= 100 and f2 < 100:
                weight1, weight2 = 0.9, 0.1
            else:
                weight1, weight2 = 0.1, 0.9
            new_genotype = []
            for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
                new_gene = weight1 * gene1 + weight2 * gene2
                new_genotype.append(new_gene)
            offspring['genotype'] = new_genotype

        else:
            # Crossover uniforme: mistura aleatória dos genes torna a procura pela solução mais rápida porque procura num espaço de resultados vasto evitando ficar preso em areas pequenas
            new_genotype = []
            for gene1, gene2 in zip(p1['genotype'], p2['genotype']):
                if random.random() < 0.5:
                    new_genotype.append(gene1)
                else:
                    new_genotype.append(gene2)
            offspring['genotype'] = new_genotype
    return offspring




def mutation1(p): #antigo -> a gaussiana é boa para pequenas alterações, facilitando a exploração local do espaço de soluções 
    mutated = copy.deepcopy(p)
    for i in range(len(mutated['genotype'])):
        if random.random() < PROB_MUTATION:
            mutated['genotype'][i] += random.gauss(0, STD_DEV)
    return mutated

def mutation2(p): # melhor que a mutação1 -> a gaussiana é boa para pequenas alterações, facilitando a exploração local do espaço de soluções 
    """Aplica mutação gaussiana ao genótipo de um indivíduo e restringe os valores entre -1 e 1."""
    mutated = copy.deepcopy(p)
    for i in range(len(mutated['genotype'])):
        if random.random() < PROB_MUTATION:
            mutated['genotype'][i] += random.gauss(0, STD_DEV)
            # Clamp: Limita o valor do gene entre -1 e 1 (pode ser ajustado conforme o problema)
            mutated['genotype'][i] = max(-1.0, min(1.0, mutated['genotype'][i]))
    return mutated

#sendo uma mutação unfiorme chega mais rapidamente aos 120 ao contrari0 da gaussiana que é mais precisa
def mutation3(p): #Aumenta a exploração global em poucos passos, permitindo que o genótipo “salte” para regiões que podem ser muito melhores, sem ficar preso em mínimos locais.
    """Aplica mutação uniforme: substitui aleatoriamente alguns genes por novos valores no intervalo [-1, 1]."""
    mutated = copy.deepcopy(p)
    for i in range(len(mutated['genotype'])):
        if random.random() < PROB_MUTATION:
            mutated['genotype'][i] = random.uniform(-1.0, 1.0)
    return mutated


def mutation4(p):
    mutated = copy.deepcopy(p)
    
    current_fitness = p.get('fitness', None)
    
    if current_fitness is not None and current_fitness >= 119:
        for i in range(len(mutated['genotype'])):
            if random.random() < PROB_MUTATION:
                # Adiciona um pequeno desvio (refinamento suave)
                mutated['genotype'][i] += random.gauss(0, STD_DEV)
                mutated['genotype'][i] = max(-1.0, min(1.0, mutated['genotype'][i]))
    else:
        for i in range(len(mutated['genotype'])):
            if random.random() < PROB_MUTATION:
                mutated['genotype'][i] = random.uniform(-3.0, 3.0)
    
    # Após a mutação, o fitness precisa ser recalcado
    mutated['fitness'] = None
    return mutated



def mutation(p):
    #Aplica mutação gaussiana para indivíduos com fitness alto, ou mutação uniforme caso contrário.
    mutated = copy.deepcopy(p)
    
    current_fitness = p.get('fitness', None)
    
    # Se o fitness for alto, usamos mutação gaussiana (refinamento)
    if current_fitness is not None and current_fitness >= 119:
        for i in range(len(mutated['genotype'])):
            if random.random() < PROB_MUTATION:
                # Adiciona um pequeno desvio (refinamento suave)
                mutated['genotype'][i] += random.gauss(0, STD_DEV)
                mutated['genotype'][i] = max(-1.0, min(1.0, mutated['genotype'][i]))
    elif current_fitness is not None and current_fitness >= 115 and current_fitness < 119:
        # Se o fitness for baixo, usamos mutação uniforme
        for i in range(len(mutated['genotype'])):
            if random.random() < PROB_MUTATION:
                mutated['genotype'][i] = random.uniform(-2.0, 2.0)
    else:
        # Se o fitness for baixo, usamos mutação uniforme (+ intervalo)
        for i in range(len(mutated['genotype'])):
            if random.random() < PROB_MUTATION:
                mutated['genotype'][i] = random.uniform(-3.0, 3.0)
    
    mutated['fitness'] = None
    return mutated






def objective_function2(observation):
    # Extrair as variáveis de estado
    x, y, vx, vy, theta, v_theta, contact_left, contact_right = observation

    # Penalizações: (velocidade + velocidade centripeta + angulo + distancia às bandeiras)
    distance_penalty = 3.0 * (abs(x) + abs(y))
    velocity_penalty = 2.0 * (abs(vx) + abs(vy))
    angle_penalty = 1.0 * (abs(theta) + 1.0 * abs(v_theta))

    # Bônus para cada perna que toca o chão
    leg_bonus = (10 if contact_left == 1 else 0) + (10 if contact_right == 1 else 0)
    
    # penalizações da distancia, da velocidade, do angulo e da velocidade centripeta mais o bônus dos pes
    fitness = - (distance_penalty + velocity_penalty + angle_penalty) + leg_bonus

    # Bonus caso o satelite pouse bem
    if check_successful_landing(observation):
        fitness += 100

    return fitness, check_successful_landing(observation)


def objective_function(observation):
    x, y, vx, vy, theta, v_theta, contact_left, contact_right = observation

    distance_to_pad = np.sqrt(x**2 + y**2)
    velocity_magnitude = np.sqrt(vx**2 + vy**2)

    distance_reward = -3.0 * distance_to_pad
    velocity_penalty = -2.0 * velocity_magnitude
    angle_penalty = -1.0 * (abs(theta) + 0.5 * abs(v_theta)) # Peso reduzido na velocidade angular

    leg_bonus = (10 if contact_left == 1 else 0) + (10 if contact_right == 1 else 0)

    landing_stability_bonus = 0
    if contact_left == 1 and contact_right == 1:
        landing_stability_bonus += 20 * np.exp(-velocity_magnitude) # Recompensa maior para velocidade menor
        landing_stability_bonus += 20 * np.exp(-abs(theta))        # Recompensa maior para ângulo menor
        landing_stability_bonus += 10 * np.exp(-abs(x)/0.2)       # Recompensa maior por estar centrado

    fitness = distance_reward + velocity_penalty + angle_penalty + leg_bonus + landing_stability_bonus

    if check_successful_landing(observation):
        fitness += 50 # Ainda um bônus significativo para sucesso total

    return fitness, check_successful_landing(observation)



    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests


if __name__ == '__main__':
    
    evolve = False
    #render_mode = 'human'
    render_mode = None
    if evolve:
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(5):    
            random.seed(seeds[i])
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    else:
        #validate individual
        bests = load_bests('log4.txt')
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
            
            
        ntests = 1000

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
        print(fit/ntests, success/ntests)

