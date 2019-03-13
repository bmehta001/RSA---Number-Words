import itertools
import math
from IPython.display import display
import pandas as pd

class NonliteralNumbersRSA:
    
    """Implementation of the core Rational Speech Acts model.

    Parameters
    ----------
    lexicon : list
        possible prices, equals states
    states : list
        possible prices
    affects : list
        possible affect choices; in this case, binary
    s_prior: list
        prior probabilities for the states
    sa_prior: list of lists
        prior probabilities for the affects given certain state
    round_cost: float
        cost for saying a round number
    sharp_cost: float
        cost for saying a non-round number
    precision: int
        rounding place value
    """
    
    # takes in lexicon = states as the possible prices, affects as the possible affect choices,
    # s_prior, sa_prior as the prior probabilities for the states and affects, round_cost,sharp_cost
    # as the costs for saying a round/non-round number and precision for rounding
    def __init__(self, lexicon, states, affects, s_prior, sa_prior, round_cost, sharp_cost,
        precision):
        assert lexicon == states, "lexicon (U) must be equal to states (S)"
        self.lexicon = lexicon
        self.states = states
        self.affects = affects
        self.meanings = list(itertools.product(self.states, self.affects))
        self.s_prior = s_prior
        self.sa_prior = sa_prior
        self.round_cost = round_cost
        self.sharp_cost = sharp_cost
        self.precision = precision # Round numbers would be to the nearest 10^precision

    # probabilities of all possible states and affects given certain utterances
    # - literal listener
    def literal_listener(self):
        # Table of L_0 for all u,s,a
        literal = [[[0 for a in self.affects] for s in self.states] for u in self.lexicon]

        for i, u in enumerate(self.lexicon):
            for j, s in enumerate(self.states):
                for k, a in enumerate(self.affects):
                    literal[i][j][k] = self.L_0(s, a, u)
        return literal
    
    # probabilities of all possible utterances given certain states, affects and goals
    # - pragmatic speaker
    def speaker(self):
        # Table of S_1 for all g,s,a,u
        speaker = [[[[0 for u in self.lexicon] for a in self.affects] for s in self.states]
            for g in self.goals()]

        for i, g in enumerate(self.goals()):
            for j, s in enumerate(self.states):
                for k, a in enumerate(self.affects):
                    for l, u in enumerate(self.lexicon):
                        # if s == 30 and a == 0 and u == 32:
                        #     import pdb; pdb.set_trace() 
                        speaker[i][j][k][l] = self.S_1(u, s, a, g)
        return speaker
    
    # probabilities of all possible states and affects given certain utterances
    # - pragmatic listener
    def listener(self):
        # Table of L_1 for all u, s, a
        listener = [[[0 for a in self.affects] for s in self.states] for u in self.lexicon]
        
        for i, u in enumerate(self.lexicon):
            for j, s in enumerate(self.states):
                for k, a in enumerate(self.affects):
                    listener[i][j][k] = self.L_1(s, a, u)
        return listener
    
    #Returns probability of certain affect being true given that a specific state is true
    def P_A(self, a, s):
        return self.sa_prior[self.states.index(s)][a]
    
    #Returns probability of certain state being true
    def P_S(self, s):
        return self.s_prior[self.states.index(s)]
    
    #Determines rounded version of utterance, based on precision
    def Round(self, x):
        return round(x, -self.precision)
    
    #Determines cost of an utterance if it is round/sharp
    def C(self, u):
        if u == self.Round(u):
            return self.round_cost
        return self.sharp_cost
    
    # Determining for a given utterance the probability 
    #  of a given state and affect
    def L_0(self, s, a, u):
        if (s == u):
            return self.P_A(a,s)
        return 0
    
    # Version of L_0 projected on a specific goal
    def L_0_projected(self, x, u, g):
        prob = 0
        for s_p, a_p in self.meanings:
            if x == g(s_p, a_p):
                prob += self.L_0(s_p, a_p, u)
        return prob

    # normalizes probability for specific state, affect, goal given utterance
    #  across utterances
    def S_1(self, u, s, a, g):
        numerator = self.S_1_helper(u, s, a, g)
        denominator = sum([self.S_1_helper(u_p, s, a, g) for u_p in self.lexicon])
        return numerator / denominator

    # normalizes probability for specific utterance given state and affect, 
    #   across states and affects
    def S_1_helper(self, u, s, a, g):
        total = 0
        for s_p, a_p in self.meanings:
            if g(s, a) == g(s_p, a_p):
                total += self.L_0(s_p, a_p, u) * math.exp(-1 * self.C(u))
        return total

    # normalizes probability for given utterance for certain state and affect
    def L_1(self, s, a, u):
        numerator = self.L_1_helper(s, a, u)
        denominator = sum([self.L_1_helper(s_p, a_p, u) for s_p in self.states for a_p in
            self.affects])
        return numerator / denominator

    # runs L_1 for all possible goals for particular state + affect to sum up 
    #   probability of an utterance
    def L_1_helper(self, s, a, u):
        total = 0
        for g in self.goals():
            total += self.P_S(s) * self.P_A(a, s) * self.P_G(g) * self.S_1(u, s, a, g)
        return total

    # Probability of a goal being likely; in this form, it is 1/6
    def P_G(self, g):
        return 1. / len(self.goals())

    # Goals iterate over f and r - 6 types
    def goals(self):
        return [lambda s, a, r=r, f=f: r(f(s), a) for r in self.generate_r() for f in
            self.generate_f()]

    # State may be itself or rounded
    def generate_f(self):
        return [lambda s: s,
                lambda s: self.Round(s)]
    
    # Types of dimensions along which goal may be determined
    def generate_r(self):
        return [lambda s, a: (s,),
                lambda s, a: (a,),
                lambda s, a: (s,a)]
                
    def display_literal_listener(self, lit):
        for u, given_u in enumerate(lit):
            lex = pd.DataFrame(given_u, index = self.states, columns = self.affects)
            d = lex.copy()
            #d['costs'] = mod.costs
            #d.loc['prior'] = list(mod.prior) + [""]
            d.loc['utterance'] = [self.lexicon[u]] + [" "]
            display(d)
            
    def display_speaker(self, speak):
        goals = ["r_{}(f_{}(s),a)".format(r,f) for r in ['s','a','sa'] for f in ['e','a']]
        for g, given_g in enumerate(speak):
            for s, given_sg in enumerate(given_g):
                lex = pd.DataFrame(given_sg, index = self.affects, columns = self.lexicon)
                d = lex.copy()
                d.loc['goal'] = [goals[g]] + [" "] + [" "]
                d.loc['state:'] = [rsa.states[s]] + [" "] + [" "]
                display(d)
                
    def display_listener(self, list):
        for u, given_u in enumerate(list):
            lex = pd.DataFrame(given_u, index = self.states, columns = self.affects)
            d = lex.copy()
            #d['costs'] = mod.costs
            #d.loc['prior'] = list(mod.prior) + [""]
            d.loc['utterance'] = [int(self.lexicon[u])] + [" "]
            display(d)
        
# states, affects, priors and other values can be modified while running the RSA class        
if __name__ == '__main__':

    # Core lexicon:

    S = U = [30, 32, 1000000]
    A = [0, 1]
    s_prior = [0.55, 0.44, 0.01]
    sa_prior = [ [0.9, 0.1],
                 [0.9, 0.1], 
                 [0.1, 0.9] ]

    rsa = NonliteralNumbersRSA(lexicon=U, states=S, affects=A, s_prior=s_prior,
                               sa_prior=sa_prior, round_cost=1, sharp_cost=3, precision=1)

        
    print("LITERAL LISTENER: ")
    rsa.display_literal_listener(rsa.literal_listener())
    
    print("SPEAKER: ")
    rsa.display_speaker(rsa.speaker())

    print("LISTENER: ")
    rsa.display_listener(rsa.listener())
