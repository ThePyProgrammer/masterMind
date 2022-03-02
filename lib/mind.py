import np
import re
import pandas as pd
from functools import reduce


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
        self.values[i] = int(obj)
        
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
            a, b = sorted(tuple(code.randSlots()))
            code[a:b] = code[a:b][::-1]
        return code
    
    def scramble(self):
        code = self.copy()
        if p() < 0.03:
            a, b = sorted(tuple(code.randSlots()))
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
        return guess.within(list(set(shield[guess != shield]) & set(guess[guess != shield]))).astype(int)
       
    def blacks(self, guess):
        return (self.values == guess.values)
    
    def score(self, guess):
        return np[self == guess, self.whites(guess).sum()]
    
    def mark(self, guess):
        output = np.full_like(self.values, ".", dtype=str)
        output[self.whites(guess)] = "W"
        output[self.blacks(guess)] = "B"
        return output
    
    def fitness(self, guesses):
        return np.array([trial.score(guess)-guess_result for (guess, guess_result) in guesses]).sum() - self.slots

    
class ChromoSet(list):
    """
    Can be used for guesses, genotype, eligibles etc
    """
    def __init__(self, values=[]):
        self.values = np.array([Chromosome(candidate) for candidate in np.array(values).tolist()], dtype=Chromosome)
        super().__init__(self.values.tolist())
        
    def __gt__(self, other):
        return len(self) > len(other)
    
    def __ge__(self, other):
        return len(self) >= len(other)
    
    def __lt__(self, other):
        return len(self) < len(other)
    
    def __le__(self, other):
        return len(self) <= len(other)
    
    def filter(self, f):
        return self.__class___([chromosome for chromosome in self if f(chromosome)])
        

class Elite(ChromoSet):
    def __add__(self, other):
        return Elite(list(set(self) | set(other)))
    def __sub__(self, other):
        return Elite(list(set(self) - set(other)))
        
        
class Genotype(ChromoSet):
    def __init__(self, size, slots, initial=[]):
        self.size = size
        self.slots = slots
        df = pd.DataFrame(initial)
        while len(df) < size[0]:
            pop = rand(size-len(df), slots)
            df = pd.concat([df, pd.DataFrame(pop)]).drop_duplicates()
        
        super().__init__(df.iloc[:size[0]].values)
        
    def reproduce(self):
        lst = []
        for i, chromosome in enumerate(self):
            if i == len(self)-1:
                lst.append(chromosome)
                break
            lst.append(chromosome.crossover(self[i+1]).mutate().invert().scramble().permute())
        
        return Genotype(len(lst), len(lst[0]), initial=list(map(Chromosome.tolist, lst)))
        
    def train(self, guesses):
        Ei = Elite()
        cnt = 0
        while h <= 100 and Ei <= self:
            children = self.reproduce().filter(lambda child: not child.fitness(guesses))
            Ei = Ei + children
            
            
            
def train(guesses):
    pop = Genotype(150, 3)
    Ei = Elite()
    cnt = 0
    while cnt <= 100 and Ei <= self:
        children = self.reproduce().filter(lambda child: not child.fitness(guesses))
        if len(children):
            Ei = Ei + children
            pop = Genotype(Ei)
    
    return Ei



        
        