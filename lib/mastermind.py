import np
import re
import pandas as pd
from collections import Counter
from functools import reduce, partial
from itertools import tee, zip_longest
import matplotlib.pyplot as plt
import seaborn as sns
import time

code_type = input("ELECTRONIC MASTERMIND 1977\nSELECT CODE TYPE: [3, 4, 5]: ").strip()
SLOTS = int(code_type)
N = 9
UNIFORM_CROSSOVER_PROBABILITY = 0.5
SINGLESTART_CROSSOVER_PROBABILITY = 0.5
SINGLEEND_CROSSOVER_PROBABILITY = 0.5
DOUBLE_CROSSOVER_PROBABILITY = 0.5
BITFLIP_PROBABILITY = 0.03
PERMUTE_PROBABILITY = 0.03
SCRAMBLE_PROBABILITY = 0.03
INVERT_PROBABILITY = 0.03
NUM_PARENTS = 50
MAX_GENERATIONS = 100
POPSIZE = 150
SLOTS = 3
ELITISM_RATE = 50
MAX_GUESS = 20
MAX_ELITES = 100


# Generates Probability Values in Array of size :param size:
def p(*size): return np.random.random(size=size if len(size) else 1)

# Generates Mastermind Values in Array of size :param size:
def rand(*size): return np.random.randint(N+1, size=size if len(size) else 1)

# Shuffles :param arr:
def shuffle(arr):
    cp = arr.copy()
    np.random.shuffle(cp)
    return cp


# Abstract Code Class
class Code(list):
    def __init__(self, *args):
        if len(args) == 1: args = args[0]
        super().__init__(map(int, args))
        self.values = np.array(self).flatten()

    @classmethod
    def randomize(cls, slots=3):
        return cls(rand(slots))

    @classmethod
    def randomDistinct(cls, size=POPSIZE, slots=SLOTS, initial=[]):
        df = pd.DataFrame(initial)
        while len(df) < size:
            pop = rand(size-len(df), slots)
            df = pd.concat([df, pd.DataFrame(pop)], ignore_index=True).drop_duplicates()
        return [cls(i) for i in df.iloc[:size].values]

    def __repr__(self):
        return f"Code("+super().__repr__()[1:-1]+")"

    def __setitem__(self, i, obj):
        super().__setitem__(i, obj)
        self.values = np.array(self)

    def randSlots(self, size=1, **kwargs):
        return np.random.choice(len(self)-1, size=size, **kwargs)

    def stringify(self):
        return "".join(map(str, self))

    @classmethod
    def destringify(cls, string):
        return cls(*map(int, string))

    def __hash__(self):
        return int(self.stringify())

    def abstractSwap(self, other=None, a=None, b=None):
        if other is None: other = self
        if a is None: a = self.randSlots()[0]
        if b is None: b = self.randSlots()[0]
        p1, p2 = map(np.array, [self, other])
        self[a], other[b] = self[b], other[a]
        return Chromosome(self), Chromosome(other)
    
    def mark(self, secret):
        guess = self.values
        secret = secret.values if hasattr(secret, "values") else np.array(secret)
        
        output = np.full_like(guess, ".", dtype=str)
        output[guess == secret] = "B"
        intersection = Counter(guess[guess != secret]) & Counter(secret[guess != secret])

        for i in np.where(guess != secret)[0]:
            if guess[i] in intersection:
                output[i] = "W"
                intersection -= Counter([guess[i]])

        return output
    
    def score(self, secret):
        guess = self.values
        secret = secret.values if hasattr(secret, "values") else np.array(secret)
        return np[(guess==secret).sum(), sum((Counter(guess[guess != secret]) & Counter(secret[guess != secret])).values())]
    
    def fitness(self, guesses):
        return np.array([np.abs(guess.score(self)-guess_result) for (guess, guess_result) in guesses]).mean()


class Chromosome(Code):
    def __init__(self, *args):
        super().__init__(*args)
        
    
    ## Crossover Methods
        
    def uniform_crossover(self, other, puniformcross=UNIFORM_CROSSOVER_PROBABILITY):
        return Chromosome(np.where(p(len(self))<puniformcross, self, other))
        
    def double_crossover(self, other, pdoublecross=DOUBLE_CROSSOVER_PROBABILITY):
        sec = slice(*sorted(self.randSlots(2,replace=False)))
        return self.abstractSwap(other, sec, sec)[0] if p() < pdoublecross else self
        
    def single_start_crossover(self, other, psinglestartcross=SINGLESTART_CROSSOVER_PROBABILITY):
        pos = self.randSlots()[0]
        return self.abstractSwap(other, slice(pos), slice(pos))[0] if p() < psinglestartcross else self
        
    def single_end_crossover(self, other, psingleendcross=SINGLEEND_CROSSOVER_PROBABILITY):
        pos = self.randSlots()[0]
        return self.abstractSwap(other, slice(pos, len(self)), slice(pos, len(self)))[0] if p() < psingleendcross else self
    
    def cross(self, other, puniformcross=UNIFORM_CROSSOVER_PROBABILITY, pdoublecross=DOUBLE_CROSSOVER_PROBABILITY, psinglestartcross=SINGLESTART_CROSSOVER_PROBABILITY, psingleendcross=SINGLEEND_CROSSOVER_PROBABILITY):
        uni = self.uniform_crossover(other, puniformcross)
        #print("Uniform", uni)
        double = uni.double_crossover(other, pdoublecross)
        #print("Double", double)
        singlestart = double.single_start_crossover(other, psinglestartcross)
        #print("Single Start", singlestart)
        singleend = singlestart.single_end_crossover(other, psingleendcross)
        #print("Single End", singleend)
        return singleend
    
    ## Mutation Methods
    
    def bitflip(self, pbitflip=BITFLIP_PROBABILITY):
        #print("bitflip length", len(self))
        return self.abstractSwap(rand(len(self)))[0]
    
    def invert(self, pinvert=INVERT_PROBABILITY):
        a, b = sorted(self.randSlots(2, replace=True))    
        return self.abstractSwap(self, slice(a,b), slice(b, a, -1))[0] if p() < pinvert else self
    
    def scramble(self, pscramble=SCRAMBLE_PROBABILITY):
        code = np.array(self)
        if p() < pscramble:
            sec = slice(*sorted(self.randSlots(2)))
            code[sec] = shuffle(code[sec])
        return Chromosome(code)
    
    def permute(self, ppermute=PERMUTE_PROBABILITY):
        return self.abstractSwap(self, *self.randSlots(2, replace=False))[0] if p() < ppermute else self
    
    def mutate(self, pbitflip=BITFLIP_PROBABILITY, pinvert=INVERT_PROBABILITY, pscramble=SCRAMBLE_PROBABILITY, ppermute=PERMUTE_PROBABILITY):
        return self.bitflip(pbitflip).invert(pinvert).scramble(pscramble).permute(ppermute)
    

class ChromoSet(list):
    def __init__(self, values=[]):
        self.values = [Chromosome(candidate) for candidate in np.array(values).tolist()]
        super().__init__(self.values)
    
    def filter(self, f):
        return Elite([chromosome for chromosome in self if f(chromosome)])
    
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
        self.slots = slots
        self.size = size
        super().__init__(Code.randomDistinct(size=size, slots=slots, initial=[]))
        
    def fitnesses(self,guesses):
        return list(map(partial(Code.fitness, guesses=guesses), self))
    
    def sortFitness(self, guesses):
        self.sort(key=partial(Code.fitness, guesses=guesses))
    
    def sortedByFitness(self, guesses):
        return sorted(self,key=partial(Code.fitness, guesses=guesses))
    
    def parents(self, guesses, n_parents=NUM_PARENTS):
        return self.sortedByFitness(guesses)[:n_parents]
        
    def breed(self, guesses, n_parents=NUM_PARENTS, elitism_rate=ELITISM_RATE):
        parents_pool = self.parents(guesses, n_parents)
        
        children = parents_pool[:elitism_rate]
        for i in range(len(self)-len(children)):
            p1 = parents_pool[int(p() * n_parents)]
            p2 = parents_pool[int(p() * n_parents)]
            #print(p1, p2)
            crossed = p1.cross(p2)
            #print("Cross", crossed)
            mutated = crossed.mutate()
            #print("Mutate", mutated)
            children.append(mutated)
        
        return Genotype(len(children), self.slots, initial=children)


code = Code.randomize(SLOTS)
init = Code.randomize(SLOTS)

print("Guess 0:", *init)
print(code.score(init), code.mark(init))
guesses = [(init, code.score(init))]

cnt = 1
    
stored = []

while cnt <= MAX_GUESS and guesses[-1][-1][0] != SLOTS:
    stored_i = len(stored)
    pop = Genotype(POPSIZE, SLOTS, [i[0] for i in guesses])
    #print(pop)
    Ei = Elite()
    gens = 1
    while gens <= MAX_GENERATIONS and len(Ei) < MAX_ELITES:
        children = pop.breed(guesses=guesses)
        eligibles = children.filter(lambda child: child.fitness(guesses) == 0)
        if len(children):
            Ei = Ei + eligibles
            pop = Genotype(POPSIZE, SLOTS, initial=Ei)
            avg_score = np.mean(pop.fitnesses(guesses))
            stored.append(avg_score)
            print("#GEN", gens, ":", avg_score)
            gens += 1
    if len(Ei) == 0: continue
    possibles = Ei - Elite([i[0] for i in guesses])
    submission = possibles[np.random.randint(len(possibles))]
    print(f"Guess {cnt}:", *submission)
    print(code.score(submission), code.mark(submission))
    guesses.append((submission, code.score(submission)))
    plt.figure(figsize=(16,8))
    plt.plot(stored[stored_i:])
    sns.regplot(x=np.arange(len(stored[stored_i:])), y=stored[stored_i:])
    plt.title(f"Guess {cnt}")
    plt.xlabel("Number of Generations")
    plt.ylabel("Average Fitness Value")
    plt.show()
    cnt += 1

print("YOU WIN!")
#return cnt
plt.figure(figsize=(20,10))
plt.plot(stored)
plt.title(f"Combined Guesses")
plt.xlabel("Number of Generations")
plt.ylabel("Average Fitness Value")
plt.show()

