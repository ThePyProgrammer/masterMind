import np
import re
import pandas as pd
from functools import reduce
from itertools import tee, zip_longest
import matplotlib.pyplot as plt
import seaborn as sns
import time
import neat

def p(*size): return np.random.random(size=size if len(size) else 1)

def rand(*size): return np.random.randint(10, size=size if len(size) else 1)

def shuffle(arr):
    cp = arr.copy()
    np.random.shuffle(cp)
    return cp

class Chromosome:
    def __init__(self, *args):
        if len(args) == 1: args = args[0]
        self.values = np.array(list(map(int, args))).flatten()
        self.slots = len(self.values)
        
    def copy(self):
        return Chromosome(*self.values.tolist())
    
    def __str__(self):
        return f"Chromosome("+", ".join(map(str, self.values))+")"
    
    def __len__(self):
        return self.slots
    
    def __getitem__(self, i):
        return self.values[i]
    
    def __setitem__(self, i, obj):
        self.values[i] = obj
        
    def randSlots(self, size=1):
        return np.random.randint(self.slots-1, size=size)
        
    def tolist(self):
        return self.values.tolist()

    def stringify(self):
        return "".join(map(str, self.tolist()))
    
    @classmethod
    def destringify(cls, string):
        return cls(*map(int, string))
    
    def crossover(self, other):
        return Chromosome(np.where(p(self.slots)<0.5, self.values, other.values))
    
    def mutate(self):
        code = self.copy()
        if p() < 0.03:
            code[code.randSlots()] = rand()
        return code
    
    def invert(self):
        code = self.copy()
        if p() < 0.03:
            a, b = sorted(tuple(code.randSlots(2)))
            code[a:b] = code[a:b][::-1]
        return code
    
    def scramble(self):
        code = self.copy()
        if p() < 0.03:
            a, b = sorted(tuple(code.randSlots(2)))
            code[a:b] = shuffle(code[a:b])
        return code
    
    def swap(self,a,b):
        self[a], self[b] = self[b], self[a]

    def permute(self):
        code = self.copy()
        for i in range(code.slots):
            if p() < 0.03: # very low
                code.swap(*code.randSlots(2))
        return code
    
    def __hash__(self):
        return int(self.stringify())
    
    def __eq__(self, other):
        return (self.values == other.values).sum()
    
    def __ne__(self, other):
        return self.values != other.values
    
    def within(self, other):
        return np.isin(self.values, other) 
    
    def whites(self, guess):
        return guess.within(list(set(self[guess != self]) & set(guess[guess != self]))).astype(int)
       
    def blacks(self, guess):
        return (self.values == guess.values)
    
    def score(self, guess):
        return np[self == guess, guess.whites(self).sum()]
    
    def mark(self, guess):
        output = np.full_like(self.values, ".", dtype=str)
        output[guess.whites(self)] = "W"
        output[guess.blacks(self)] = "B"
        return output
    
    def markWith(self, optimal):
        guess = self.values.copy()+1
        code = optimal.values.copy()+1
        output = np.full_like(code, ".", dtype=str)
        output[guess == code] = "B"
        negatize = code * ((code != guess)*2 - 1)
        considerations = []

        for allele in guess[guess != code]:
            result = "."
            if allele in negatize:
                result = "W"
                negatize[np.argmax(negatize == allele)] = -allele
            considerations.append(result)

        output[guess != code] = considerations

        return output
    
    def scoreWith(self, optimal):
        output = self.markWith(optimal)
        return np[(output == "B").sum(), (output == "W").sum()]
    
    def fitness(self, guesses):
        return np.abs(np.array([self.scoreWith(guess)-guess_result for (guess, guess_result) in guesses])).sum() #- self.slots

    
class ChromoSet(list):
    """
    Can be used for guesses, genotype, eligibles etc
    """
    def __init__(self, values=[]):
        self.values = [Chromosome(candidate) for candidate in np.array(values).tolist()]
        super().__init__(self.values)
        
    def __gt__(self, other):
        return len(self) > len(other)
    
    def __ge__(self, other):
        return len(self) >= len(other)
    
    def __lt__(self, other):
        return len(self) < len(other)
    
    def __le__(self, other):
        return len(self) <= len(other)
    
    def filter(self, f):
        return Elite([chromosome for chromosome in self if f(chromosome)])
    
    def tolist(self):
        return list(map(Chromosome.tolist, self))
        

class Elite(ChromoSet):
    def __add__(self, other):
        return Elite(list(set(self) | set(other)))
    def __sub__(self, other):
        return Elite(list(set(self) - set(other)))
    def __str__(self):
        return "["+", ".join(map(str, self))+"]"
    def __repr__(self):
        return "["+", ".join(map(str, self))+"]"
        
        
class Genotype(ChromoSet):
    def __init__(self, size, slots, initial=[]):
        self.size = size
        self.slots = slots
        df = pd.DataFrame(initial)
        while len(df) < size:
            pop = rand(size-len(df), slots)
            df = pd.concat([df, pd.DataFrame(pop)], ignore_index=True).drop_duplicates()
        
        super().__init__(df.iloc[:size].values)
        
    def reproduce(self):
        a, b = tee(self)
        next(b, None)
        return Genotype(len(self), len(self[0]), initial=list(map(lambda chromosomes: chromosomes[0].crossover(chromosomes[1]).mutate().invert().scramble().permute().tolist(), zip_longest(a, b, fillvalue=self[-1]))))


def run_test(N):
    def train(guesses):
        pop = Genotype(150, N)
        Ei = Elite()
        cnt = 0
        while cnt <= 100 and Ei <= pop:
            children = pop.reproduce().filter(lambda child: not child.fitness(guesses))
            if len(children):
                Ei = Ei + children
                pop = Genotype(150, N, initial=list(map(Chromosome.tolist, Ei)))
            cnt += 1

        return Ei

    code = Chromosome(rand(N))
    init = Chromosome(rand(N))

    print("Guess 0:", *init)
    print(init.scoreWith(code), init.markWith(code))
    guesses = [(init, init.scoreWith(code))]

    cnt = 1

    while cnt <= 100 and guesses[-1][-1][0] != N:
        Ei = train(guesses)
        if len(Ei) == 0: continue
        possibles = Ei - Elite([i[0].tolist() for i in guesses])
        #print("Elites:",possibles)
        submission = possibles[np.random.randint(len(possibles))]
        print(f"Guess {cnt}:", *submission)
        print(submission.scoreWith(code), submission.markWith(code))
        guesses.append((submission, submission.scoreWith(code)))
        cnt += 1

    print("YOU WIN!")
    return cnt



def run(N, config_path="config.txt"):
    code = Chromosome(rand(N))
    init = Chromosome(rand(N))

    print("Guess 0:", *init)
    print(init.scoreWith(code), init.markWith(code))
    guesses = [(init, init.scoreWith(code))]

    def eval_genomes(genomes, config): 
        pop = Genotype(len(genomes))
        ge = []

        for (genome_id, genome), chromosome in zip(genomes, pop.tolist()):
            ge.append(genome)
            genome.fitness = 20
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = chromosome.fitness(guesses)
            
    config = neat.config.Config(  # We first configure the neat configuration
        neat.DefaultGenome,       # Add the necessary arguments (defaults)
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path               # Add file path to the config file
    )

    pop = neat.Population(config) # Add the population of dinosaurs
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    
    
    winner = pop.run(eval_genomes, 300)     # Run the evolution function 50 times 

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    
    
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)

    



























def run(config_path):
    config = neat.config.Config(  # We first configure the neat configuration
        neat.DefaultGenome,       # Add the necessary arguments (defaults)
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path               # Add file path to the config file
    )

    pop = neat.Population(config) # Add the population of dinosaurs
    pop.run(eval_genomes, 50)     # Run the evolution function 50 times 

run_test(3)
