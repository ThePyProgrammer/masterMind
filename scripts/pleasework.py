import np, re, random, time
from collections import Counter
import matplotlib.pyplot as plt

def GA(
    SLOTS=3,
    N = 9,
    UNIFORM_CROSSOVER_PROBABILITY = 0.5,
    SINGLESTART_CROSSOVER_PROBABILITY = 0,
    SINGLEEND_CROSSOVER_PROBABILITY = 0,
    DOUBLE_CROSSOVER_PROBABILITY = 1,
    BITFLIP_PROBABILITY = 0.1,
    PERMUTE_PROBABILITY = 0.03,
    SCRAMBLE_PROBABILITY = 0.03,
    INVERT_PROBABILITY = 0.03,
    NUM_PARENTS = 20,
    MAX_GENERATIONS = 50,
    POPSIZE = 100,
    ELITISM_RATE = 30,
    MAX_GUESS = 20,
    MAX_ELITES = 10,
    a = 1,
    b = 1,
    show_printout=True
):

    if show_printout: print("ELECTRONIC MASTERMIND 1977")
    if show_printout: print("You have selected", SLOTS, "pegs.")


    # Generates Probability Values in Array of size :param size:
    p = random.random

    # Generates Mastermind Values in Array of size :param size:
    def rand(*size): return np.random.randint(N+1, size=size if len(size) else 1)


    # Abstract Code Class
    class Code(list):
        def __init__(self, *args):
            if len(args) == 1: args = args[0]
            super().__init__(map(int, args))

        @classmethod
        def randomize(cls, slots=3):
            return cls(rand(slots))

        @classmethod
        def randomDistinct(cls, size=POPSIZE, slots=SLOTS, initial=[]):
            pop = set(map(cls, initial))
            while len(pop) < size:
                pop = pop & set(map(cls, rand(size-len(pop), slots)))
            return list(pop)

        def __repr__(self):
            return f"Code("+super().__repr__()[1:-1]+")"
        
        def __eq__(self, other):
            return hash(self) == hash(other)

        def stringify(self):
            return "".join(map(str, self))

        def __hash__(self):
            return hash(self.stringify())
        
        def mark(self, secret):
            output = ["." for i in range(SLOTS)]
            bcount = wcount = 0
            for i in range(len(self)):
                if self[i] == secret[i]:
                    output[i] = "B"
                    bcount += 1
            
            intersection = Counter([secret[i] for i in range(SLOTS) if output[i] != "B"]) & Counter([self[i] for i in range(SLOTS) if output[i] != "B"])

            for i in range(len(self)):
                if output[i] == "B": continue
                if self[i] in intersection and intersection[self[i]] > 0:
                    output[i] = "W"
                    wcount += 1
                    intersection[self[i]] -= 1

            return (bcount, wcount), output
        
        def fitness(self, guesses):
            sum = 0
            for (guess, guess_result) in guesses:
                B, W = guess.mark(self)[0]
                sum += a*abs(B - guess_result[0]) + b*abs(W - guess_result[1])
            return sum/len(guesses)


    class Chromosome(Code):
        def __init__(self, *args):
            super().__init__(*args)
            
        
        ## Crossover Methods
            
        def uniform_crossover(self, other, puniformcross=UNIFORM_CROSSOVER_PROBABILITY):
            return [self[i] if p() > puniformcross else other[i] for i in range(SLOTS)]
            
        def double_crossover(self, other, pdoublecross=DOUBLE_CROSSOVER_PROBABILITY):
            code = self.copy()
            if p() < pdoublecross:
                a, b = sorted(np.random.choice(SLOTS, size=2, replace=False))
                code[a:b] = other[a:b]
            return code
            
        def single_start_crossover(self, other, psinglestartcross=SINGLESTART_CROSSOVER_PROBABILITY):
            code = self.copy()
            if p() < psinglestartcross:
                pos = random.randint(0, SLOTS-1)
                self[pos:] = other[pos:]
            return code
            
        def single_end_crossover(self, other, psingleendcross=SINGLEEND_CROSSOVER_PROBABILITY):
            code = self.copy()
            if p() < psingleendcross:
                pos = random.randint(0, SLOTS-1)
                self[:pos] = other[:pos]
            return code
        
        def cross(self, other, puniformcross=UNIFORM_CROSSOVER_PROBABILITY, pdoublecross=DOUBLE_CROSSOVER_PROBABILITY, psinglestartcross=SINGLESTART_CROSSOVER_PROBABILITY, psingleendcross=SINGLEEND_CROSSOVER_PROBABILITY):
            uni = Chromosome.uniform_crossover(self, other, puniformcross)
            double = Chromosome.double_crossover(uni, other, pdoublecross)
            singlestart = Chromosome.single_start_crossover(double, other, psinglestartcross)
            singleend = Chromosome.single_end_crossover(singlestart, other, psingleendcross)
            return singleend
        
        ## Mutation Methods
        
        def bitflip(self, pbitflip=BITFLIP_PROBABILITY):
            code = self.copy()
            if p() < pbitflip:
                code[random.randint(0, SLOTS-1)] = rand()
            return code
        
        def invert(self, pinvert=INVERT_PROBABILITY):
            code = self.copy()
            if p() < pinvert:
                a, b = sorted(np.random.choice(SLOTS, size=2, replace=False))
                code =  code[:a]+code[b:a:-1]+code[b:]
            return code
        
        def scramble(self, pscramble=SCRAMBLE_PROBABILITY):
            code = self.copy()
            if p() < pscramble:
                a, b = sorted(np.random.choice(SLOTS, size=2, replace=False))
                code[a:b] = np.random.permutation(code[a:b])
            return code
        
        def permute(self, ppermute=PERMUTE_PROBABILITY):
            out = self.copy()
            if p() < ppermute:
                a, b = sorted(np.random.choice(SLOTS, size=2, replace=False))
                out[a], out[b] = out[b], out[a]
            return out
            
        def mutate(self, pbitflip=BITFLIP_PROBABILITY, pinvert=INVERT_PROBABILITY, pscramble=SCRAMBLE_PROBABILITY, ppermute=PERMUTE_PROBABILITY):
            flips = Chromosome.bitflip(self, pbitflip)
            inverts = Chromosome.invert(flips, pinvert)
            scrambled = Chromosome.scramble(inverts, pscramble)
            return Chromosome.permute(scrambled, ppermute)
        

    class Genotype(list):
        def __init__(self, size, slots, initial=[], guesses_list=None):
            if guesses_list is None: guesses_list = guesses
            self.slots = slots
            pop = set(initial)
            while len(pop) < size:
                pop |= {Chromosome([random.randint(0,9) for i in range(slots)]) for i in range(size-len(pop))}
            
            self.values = [(i, i.fitness(guesses_list)) for i in pop]
            #print(self.fitnesses)
            self.values.sort(key=lambda a:a[1])
            super().__init__([i[0] for i in self.values])
            self.fitnesses = [i[1] for i in self.values]
        
        def parents(self, n_parents=NUM_PARENTS):
            return self[:n_parents]
        
        def eligibles(self):
            index = -1
            for f in self.fitnesses:
                if f > 0: break
                index += 1
            
            return self[:index+1]
            
        def breed(self, guesses, n_parents=NUM_PARENTS, elitism_rate=ELITISM_RATE):
            parents_pool = self.parents(n_parents)
            
            children = parents_pool[:elitism_rate]
            for i in range(len(self)-len(children)):
                a, b = np.random.choice(n_parents, size=2, replace=False)
                crossed = parents_pool[a].cross(parents_pool[b])
                mutated = Chromosome.mutate(crossed)
                children.append(Chromosome(mutated))
            
            return Genotype(len(self), self.slots, initial=children, guesses_list=guesses) #, sum([i[1] for i in self.fitnesses])/150



    code = Code.randomize(SLOTS)

    if show_printout: print("Code is", *code)
    init = Code(range(SLOTS))

    if show_printout: print("Guess 1:", *init)
    score, marked = code.mark(init)
    if show_printout: print(*score, marked)
    guesses = [(init, score)]
    guessed_codes = [init]

    cnt = 2

    stored = []
    avg_scores = []

    start = time.time()

    while cnt <= MAX_GUESS and guesses[-1][-1][0] != SLOTS:
        stored_i = len(stored)
        start_guess = time.time()
        pop = Genotype(POPSIZE, SLOTS, guesses_list=guesses) # guessed_codes, 
        #print(pop)
        Ei = set()
        gens = 1

        while gens <= MAX_GENERATIONS and len(Ei) < MAX_ELITES:
            gen_time = time.time()
            children = pop.breed(guesses=guesses)
            Ei = Ei.union(pop.eligibles(), children.eligibles())
            avg_score = (sum(pop.fitnesses) + sum(children.fitnesses))/(2*POPSIZE)
            pop = Genotype(POPSIZE, SLOTS, initial=Ei, guesses_list=guesses)
            
            stored.append(avg_score)
            if show_printout: print("#GEN", gens, ":", avg_score)
            if show_printout: print("Elapsed Time of Generation", time.time()-gen_time)
            gens += 1

        if len(Ei):
            possibles = tuple(Ei - set(guessed_codes))
            submission = possibles[int(p() * len(possibles))]
        else:
            continue
        if show_printout: print(f"Guess {cnt}:", *submission)
        if show_printout: print("Elapsed Time of Guess:", time.time()-start_guess)


        score, marked = code.mark(submission)
        if show_printout: print(score, marked)
        guesses.append((submission, score))
        avg_scores.append(stored[stored_i:])
        cnt += 1

    if show_printout: print("YOU WIN!")
    
    elapsed_time = time.time() - start
    if show_printout:
        print("Time elapsed was:", elapsed_time)

        for i,f in enumerate(avg_scores):
            plt.figure(figsize=(16,8))
            plt.plot(f)
            plt.title(f"Guess {i+2}")
            plt.xlabel("Number of Generations")
            plt.ylabel("Average Fitness Value")
            plt.show()


        #return cnt
        plt.figure(figsize=(20,10))
        plt.plot(stored)
        plt.title(f"Combined Guesses")
        plt.xlabel("Number of Generations")
        plt.ylabel("Average Fitness Value")
        plt.show()
    
    return len(guesses), elapsed_time, len(stored)
    
    



#code_type = input("SELECT CODE TYPE: [3, 4, 5]: ").strip()
SLOTS = random.randint(3,5)#int(code_type)
GA(SLOTS=SLOTS)
