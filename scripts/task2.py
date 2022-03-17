from random import randint, random, sample, choice, choices
from time import time
from math import floor
import enum

'''
Crossover type enum
-one_point for single point crossover
-two_point for double point crossover
'''
class crossover_type(enum.Enum):
    one_point = 0
    two_point = 1

'''
Selection type enum
-roulette_wheel for roulette wheel selection
-tournament_selection for k tournament_selection
'''
class selection_type(enum.Enum):
    roulette_wheel = 0
    tournament_selection = 1


show_fitness = False
code_length = 5
population_size = 50
mutation_rate = 0.1
permutation_rate = 0.03
inversion_rate = 0.02
crossover_rate = 0.8
elitism_rate = 0
max_gen = 30
max_eligible = 10
crossover_method = crossover_type.one_point
selection_method = selection_type.roulette_wheel
k = -1
if(k < 1 or k > population_size): k = floor(population_size * 0.2)

fitness_func_a = 1
fitness_func_b = 0

guess_list = []
output_list = []
eligible_bag = set()
solution = [randint(0,9) for i in range(code_length)]
guess_count = 0
total_generations = 0
avg_fitness_lists = []
    
print("ELECTRONIC MASTERMIND 1977 GENETIC ALGORITHM SOLVER")
print("Population size:", population_size)
print("Max Generation:", max_gen)
print("Mutation rate:", mutation_rate)
print("Inversion rate:", inversion_rate)
print("Permutation rate:", permutation_rate)
print("Elitism:", bool(elitism_rate))
print("Elitism rate:", elitism_rate)
print("code type:",code_length, end="\n\n")

start_time_global = time()

#doing the initial guess [0,1,2], [0,1,2,3] or [0,1,2,3,4]
initial_guess = [i for i in range(code_length)]
print("GA:", initial_guess)
guess_list.append(initial_guess)
guess_count += 1

#calculating the intial output

def pick_one(population, 
             probability_distribution):
    if(selection_method == selection_type.roulette_wheel):
        #invert probability disribution to bias results with the smallest scores
        new_prob_dis = [max(probability_distribution)*1.01 - i for i in probability_distribution]
        return choices(population, new_prob_dis)[0]
    elif(selection_method == selection_type.tournament_selection):
        k_competitors = sample(list(zip(population, probability_distribution)), k)
        best_chromosome = min(k_competitors, key = lambda x : x[1])
        return best_chromosome[0]




def comparison(guess, solution):
    output = ["." for i in range(code_length)]
    for i in range(code_length):
        if solution[i] == guess[i]:
            output[i] = 'B'
        else:
            for j in range(code_length):
                if (i != j) and (solution[i] == guess[j]):
                    if output[j] == '.':
                        output[j] = 'W'
                        break
    return output

def fitness_function(chromosome):
    is_eligible = False
    fitness = 0
    for i in range(guess_count):
        output_chromosome_guess = comparison(chromosome, guess_list[i])
        fitness += abs(fitness_func_a * (output_chromosome_guess.count("B") - output_list[i].count("B")))
        fitness += abs(output_chromosome_guess.count("W") - output_list[i].count("W"))
    if (fitness == 0):
        is_eligible = True
    fitness += fitness_func_b * code_length * (guess_count)
    return [is_eligible, fitness]

def evaluate_population_fitness(population):
    score = []
    eligible_bag_temp = set()
    for i in population:
        is_eligible, fitness_value = fitness_function(i)
        score.append(fitness_value)
        if is_eligible:
            eligible_bag_temp.add("".join(map(str,i)))
    return [eligible_bag_temp, score]



#crossover
#one point or two point crossover
def crossover(parent1, parent2):
    if(random() < crossover_rate):
        child1 = []
        child2 = []
        if(crossover_method == crossover_type.one_point):
            point = randint(1, code_length - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
        elif(crossover_method == crossover_type.two_point):
            point1, point2 = sorted(sample([i for i in range(1, code_length)], 2))
            child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return choice([child1, child2])
    else:
        return choice([parent1, parent2])

#mutation
def mutation(chromosome):
    chromosome = chromosome.copy()
    if(random() < mutation_rate):
        chromosome[randint(1, code_length - 1)] = randint(0,9)
    return chromosome

#inversion
def inversion(chromosome):
    chromosome = chromosome.copy()
    if(random() < inversion_rate):
        point1, point2 = sorted(sample([i for i in range(0, code_length+1)], 2))
        chromosome = chromosome[:point1] + chromosome[point1:point2][::-1] + chromosome[point2:]
    return chromosome

#permutation
def permutation(chromosome):
    chromosome = chromosome.copy()
    if(random() < inversion_rate):
        point1, point2 = sorted(sample([i for i in range(0, code_length)], 2))
        chromosome[point1], chromosome[point2] = chromosome[point2], chromosome[point1]
    return chromosome






    
    
initial_output = comparison(initial_guess, solution)
print("Guess", str(guess_count) + ":", initial_output.count("B"), initial_output.count("W"), initial_output)
output_list.append(initial_output)

#unsolved_flag for game_loop
unsolved_flag = True

#main game loop
while(unsolved_flag):
    #generating the first population for a guess
    avg_fitness_list = []
    start_time = time()
    first_population = [[randint(0,9) for i in range(code_length)] for i in range(population_size)]
    eligible_bag, first_population_fitness = evaluate_population_fitness(first_population)
    end_time = time()
    if show_printout: 
        print("\nGEN # 1")
        print("Average Fitness:", sum(first_population_fitness)/len(first_population_fitness))
        print("Elapsed Time:", end_time - start_time)
    avg_fitness_list.append(sum(first_population_fitness)/len(first_population_fitness))

    #calculating the global best
    global_best_index = min(zip(first_population_fitness, range(len(first_population_fitness))))[1]
    global_best = first_population[global_best_index]
    global_best_score = first_population_fitness[global_best_index]

    #placing the list in populations
    populations = [first_population]
    scores = [first_population_fitness]
    gen_count = 1

    #performing GA tasks
    while(gen_count < max_gen and len(eligible_bag) < max_eligible):  

        #increasing generation count and retrieving previous populations
        gen_count += 1
        previous_population = populations[-1]
        previous_scores = scores[-1]

        #getting the population size and creating the new_population list
        start_time = time()
        new_population = []
        #local_population_size is the size of the population whcih will undergo crossover
        local_population_size = population_size

        #selecting the elites
        #number of elites is equal to the local population size multiplied by the elitism rate
        if(elitism_rate):
            previous_scores, previous_population = zip(*sorted(zip(previous_scores, previous_population)))
            previous_scores = list(previous_scores)
            previous_population = list(previous_population)
            number_of_elites = floor(local_population_size * elitism_rate)
            new_population = previous_population[:number_of_elites]

            #decreasing the size of the population which will undergo crossover
            local_population_size -= number_of_elites
            previous_population = previous_population[number_of_elites:]
            previous_scores = previous_scores[number_of_elites:]

        #crossover
        for i in range(local_population_size):
            parent1 = pick_one(previous_population, previous_scores).copy()
            parent2 = pick_one(previous_population, previous_scores).copy()
            child = crossover(parent1, parent2)
            mutated_child = inversion(permutation(mutation(child)))
            new_population.append(mutated_child)

        #evaluation of population fitness
        eligible_bag_temp, new_population_fitness = evaluate_population_fitness(new_population)

        # picking the best chromosome and storing them in the case of no eligible options
        local_best_index = min(zip(new_population_fitness, range(len(new_population_fitness))))[1]
        if(new_population_fitness[local_best_index] < global_best_score):
            global_best = new_population[local_best_index]
            global_best_score = new_population_fitness[local_best_index]

        #adding the new eligible chromosomes to the total bag
        eligible_bag = eligible_bag.union(eligible_bag_temp)
        end_time = time()
        if show_printout:
            print("\nGEN #", gen_count)
            print("Average Fitness:", sum(new_population_fitness)/len(new_population_fitness))
            print("Elapsed Time:", end_time - start_time)
        avg_fitness_list.append(sum(new_population_fitness)/len(new_population_fitness))

        #adding the new populations to the population list
        populations.append(new_population)
        scores.append(new_population_fitness)

    avg_fitness_lists.append(avg_fitness_list)
    total_generations += gen_count

    #the new guess is the global best or a random eligible chromsome
    if(eligible_bag):
        new_guess = [int(i) for i in sample(eligible_bag, 1)[0]]
    else:
        new_guess = global_best

    #guessing
    if show_printout: print("\nGA:", new_guess)
    guess_list.append(new_guess)
    guess_count += 1
    new_output = comparison(new_guess, solution)
    if show_printout: print("Guess", str(guess_count) + ":", new_output.count("B"), new_output.count("W"), new_output)
    output_list.append(new_output)

    #checking if solved
    if(new_output.count("B") == code_length):
        unsolved_flag = False
if show_printout: print("SOLVED:", new_guess)
if show_fitness:

    fig, ax = plt.subplots(guess_count - 1,figsize=(4,5 * (guess_count - 1)))
    for i in range(guess_count - 1):
        ax[i].plot(avg_fitness_lists[i])
        ax[i].title.set_text("Guess " + str(i+2))



total_time = time() - start_time_global

                   
        #Main printout
        if (show_printout):
            print("ELECTRONIC MASTERMIND 1977 GENETIC ALGORITHM SOLVER")
            print("Population size: " + str(population_size))
            print("Max Generation: " + str(max_gen))
            print("Mutation rate: " + str(mutation_rate))
            print("Inversion rate: " + str(inversion_rate))
            print("Permutation rate: " + str(permutation_rate))
            print("Elitism: " + str(bool(elitism_rate)))
            print("Elitism rate: " + str(elitism_rate))
            print("code type: " + str(code_length) + "\n")
            
        #game loop
        start_time_global = time()
        game_loop()
        total_time = time() - start_time_global
        
    #converts the chromosome to a string to make it a hashable type
    def as_string(self,
                  chromosome):
        return "".join([str(i) for i in chromosome])
    
    #converts the string of a chromosome back to a chromosome, making it unhashable
    def as_list(self,
                chromosome_string):
        return [int(i) for i in chromosome_string]
    
    #This function is used to get the output code e.g. [".", ".", ".", "B", "W"]
    def comparison(self,
                   guess,
                   solution):
        output = ["." for i in range(code_length)]
        for i in range(code_length):
            if solution[i] == guess[i]:
                output[i] = 'B'
            else:
                for j in range(code_length):
                    if (i != j) and (solution[i] == guess[j]):
                        if output[j] == '.':
                            output[j] = 'W'
                            break
        return output
    
    #fitness function
    #returns eligibility(based on the paper), fitness for a given chromosome
    def fitness_function(self,
                         chromosome):
        is_eligible = False
        fitness = 0
        for i in range(guess_count):
            output_chromosome_guess = comparison(chromosome, guess_list[i])
            fitness += abs(fitness_func_a * (output_chromosome_guess.count("B") - output_list[i].count("B")))
            fitness += abs(output_chromosome_guess.count("W") - output_list[i].count("W"))
        if (fitness == 0):
            is_eligible = True
        fitness += fitness_func_b * code_length * (guess_count)
        return [is_eligible, fitness]
    
    #crossover
    #one point or two point crossover
    def crossover(self,
                  parent1,
                  parent2):
        if(random() < crossover_rate):
            child1 = []
            child2 = []
            if(crossover_method == crossover_type.one_point):
                point = randint(1, code_length - 1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
            elif(crossover_method == crossover_type.two_point):
                point1, point2 = sorted(sample([i for i in range(1, code_length)], 2))
                child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
                child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
            return choice([child1, child2])
        else:
            return choice([parent1, parent2])
    
    #mutation
    def mutation(self,
                 chromosome):
        chromosome = chromosome.copy()
        if(random() < mutation_rate):
            chromosome[randint(1, code_length - 1)] = randint(0,9)
        return chromosome
    
    #inversion
    def inversion(self,
                  chromosome):
        chromosome = chromosome.copy()
        if(random() < inversion_rate):
            point1, point2 = sorted(sample([i for i in range(0, code_length+1)], 2))
            chromosome = chromosome[:point1] + chromosome[point1:point2][::-1] + chromosome[point2:]
        return chromosome
    
    #permutation
    def permutation(self,
                    chromosome):
        chromosome = chromosome.copy()
        if(random() < inversion_rate):
            point1, point2 = sorted(sample([i for i in range(0, code_length)], 2))
            chromosome[point1], chromosome[point2] = chromosome[point2], chromosome[point1]
        return chromosome
    
    #function to evealuate an entire populations fitness
    def evaluate_population_fitness(self, population):
        score = []
        eligible_bag_temp = set()
        for i in population:
            is_eligible, fitness_value = fitness_function(i)
            score.append(fitness_value)
            if is_eligible:
                eligible_bag_temp.add(as_string(i))
        return [eligible_bag_temp, score]
    
    #selection
    #roulette wheel or tournament selection
    def pick_one(self, 
                 population, 
                 probability_distribution):
        if(selection_method == selection_type.roulette_wheel):
            #invert probability disribution to bias results with the smallest scores
            new_prob_dis = [max(probability_distribution)*1.01 - i for i in probability_distribution]
            return choices(population, new_prob_dis)[0]
        elif(selection_method == selection_type.tournament_selection):
            k_competitors = sample(list(zip(population, probability_distribution)), k)
            best_chromosome = min(k_competitors, key = lambda x : x[1])
            return best_chromosome[0]
            
    #main game loop
    def game_loop(self):
        #doing the initial guess [0,1,2], [0,1,2,3] or [0,1,2,3,4]
        initial_guess = [i for i in range(0, code_length)]
        if show_printout: print("GA:", initial_guess)
        guess_list.append(initial_guess)
        guess_count += 1
        
        #calculating the intial output
        initial_output = comparison(initial_guess, solution)
        if show_printout: print("Guess", str(guess_count) + ":", initial_output.count("B"), initial_output.count("W"), initial_output)
        output_list.append(initial_output)
        
        #unsolved_flag for game_loop
        unsolved_flag = True
        
        #main game loop
        while(unsolved_flag):
            
            #generating the first population for a guess
            avg_fitness_list = []
            start_time = time()
            first_population = [[randint(0,9) for i in range(code_length)] for i in range(population_size)]
            eligible_bag, first_population_fitness = evaluate_population_fitness(first_population)
            end_time = time()
            if show_printout: 
                print("\nGEN # 1")
                print("Average Fitness:", sum(first_population_fitness)/len(first_population_fitness))
                print("Elapsed Time:", end_time - start_time)
            avg_fitness_list.append(sum(first_population_fitness)/len(first_population_fitness))
            
            #calculating the global best
            global_best_index = min(zip(first_population_fitness, range(len(first_population_fitness))))[1]
            global_best = first_population[global_best_index]
            global_best_score = first_population_fitness[global_best_index]
            
            #placing the list in populations
            populations = [first_population]
            scores = [first_population_fitness]
            gen_count = 1
            
            #performing GA tasks
            while(gen_count < max_gen and len(eligible_bag) < max_eligible):  
                
                #increasing generation count and retrieving previous populations
                gen_count += 1
                previous_population = populations[-1]
                previous_scores = scores[-1]
                
                #getting the population size and creating the new_population list
                start_time = time()
                new_population = []
                #local_population_size is the size of the population whcih will undergo crossover
                local_population_size = population_size
                
                #selecting the elites
                #number of elites is equal to the local population size multiplied by the elitism rate
                if(elitism_rate):
                    previous_scores, previous_population = zip(*sorted(zip(previous_scores, previous_population)))
                    previous_scores = list(previous_scores)
                    previous_population = list(previous_population)
                    number_of_elites = floor(local_population_size * elitism_rate)
                    new_population = previous_population[:number_of_elites]
                    
                    #decreasing the size of the population which will undergo crossover
                    local_population_size -= number_of_elites
                    previous_population = previous_population[number_of_elites:]
                    previous_scores = previous_scores[number_of_elites:]
                
                #crossover
                for i in range(local_population_size):
                    parent1 = pick_one(previous_population, previous_scores).copy()
                    parent2 = pick_one(previous_population, previous_scores).copy()
                    child = crossover(parent1, parent2)
                    mutated_child = inversion(permutation(mutation(child)))
                    new_population.append(mutated_child)
                
                #evaluation of population fitness
                eligible_bag_temp, new_population_fitness = evaluate_population_fitness(new_population)
            
                # picking the best chromosome and storing them in the case of no eligible options
                local_best_index = min(zip(new_population_fitness, range(len(new_population_fitness))))[1]
                if(new_population_fitness[local_best_index] < global_best_score):
                    global_best = new_population[local_best_index]
                    global_best_score = new_population_fitness[local_best_index]
                
                #adding the new eligible chromosomes to the total bag
                eligible_bag = eligible_bag.union(eligible_bag_temp)
                end_time = time()
                if show_printout:
                    print("\nGEN #", gen_count)
                    print("Average Fitness:", sum(new_population_fitness)/len(new_population_fitness))
                    print("Elapsed Time:", end_time - start_time)
                avg_fitness_list.append(sum(new_population_fitness)/len(new_population_fitness))
                
                #adding the new populations to the population list
                populations.append(new_population)
                scores.append(new_population_fitness)
            
            avg_fitness_lists.append(avg_fitness_list)
            total_generations += gen_count
            
            #the new guess is the global best or a random eligible chromsome
            if(eligible_bag):
                new_guess = as_list(sample(eligible_bag, 1)[0])
            else:
                new_guess = global_best
            
            #guessing
            if show_printout: print("\nGA:", new_guess)
            guess_list.append(new_guess)
            guess_count += 1
            new_output = comparison(new_guess, solution)
            if show_printout: print("Guess", str(guess_count) + ":", new_output.count("B"), new_output.count("W"), new_output)
            output_list.append(new_output)
            
            #checking if solved
            if(new_output.count("B") == code_length):
                unsolved_flag = False
        if show_printout: print("SOLVED:", new_guess)
        if show_fitness:
            
            fig, ax = plt.subplots(guess_count - 1,figsize=(4,5 * (guess_count - 1)))
            for i in range(guess_count - 1):
                ax[i].plot(avg_fitness_lists[i])
                ax[i].title.set_text("Guess " + str(i+2))
                