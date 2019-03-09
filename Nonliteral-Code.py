
# S = [30, 32, 1000000]
# A = [0,1]
# G = ['s','a','sa']
# priorAffects = [[0,1],[0,0],[0,0]]
# sa = list(itertools.product(s,a))
# print(sa)

# roundCost = 1
# sharpCost = 0
# digitsOfPrecision = 1

# def Round(x):
#     return round(x,digitsOfPrecision-len(str(x)))
    
# # def L_0(s,a,u):
# #     if (s == u):
# #         return P_A(a,s)
# #     else:
# #         return 0

# def P_A(a,s):
#     return priorAffects[allS.index(s)][a]

# def g(s,a):
#     return ((s),(a),(s,a))

# def L_0(x,u,g):
#     cumul = 0
#     for s_p, a_p in sa:
#         if x == g(s_p, a_p):
#             cumul+=L_0(s_p, a_p, u)
#     return cumul



# def priorStates(possStates, possUtterances): 
#     prior = []
#     for x in possUtterances:
#         for y in possStates:
#             prior.append()

    

# def cost(u):
#     if u%10 == 0:
#         return roundCost
#     return sharpCost

import itertools

class NonliteralNumbersRSA:
    
    # TODO: Explain parameters
    def __init__(self, lexicon, states, affects, s_prior, sa_prior, round_cost, sharp_cost, precision=1):
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
    
    def listeral_listener(self):
        # TODO: Should display table of L_0 for all s,a
        pass
    
    def speaker(self):
        # TODO: Should display table of S_1 for all u,s,a,g.
        # This will be a lot of output, so we could do this by maybe having
        # optional parameters to this function to narrow the output.
        pass
    
    def listener(self):
        # TODO: Should display table of L_1 for all s,a,u
        pass
    
    def P_A(self, a, s):
        return self.sa_prior[self.states.index(s)][a]
    
    def P_S(self, s):
        return self.s_prior[self.states.index(s)]
    
    def Round(x):
        return round(x, -precision)
    
    def C(self, u):
        if u == self.Round(u):
            return self.round_cost
        return sharp_cost
    
    def L_0(self, s, a, u):
        if (s == u):
            return self.P_A(a,s)
        return 0
    
    def L_0_projected(self, x, u, g):
        prob = 0
        for s_p, a_p in self.meanings:
            if x == g(s_p, a_p):
                prob += self.L_0(s_p, a_p, u)
        return prob
    
    def S_1(self, u, s, a, g):
        # TODO: This should implement equation [8], making sure to normalize over
        # all s,a,g since it's a conditional probability
        pass
    
    def L_1(self, s, a, u):
        # TODO: This should implement equation [10], making sure to normalize over
        # all u since it's a conditional probability.
        # To iterate over goal functions, use 
        # for g in self.goals().  (each g will be a function)
    
    def goals(self):
        return [lambda s, a: r(f(s), a) for f in generate_f for r in generate_r]

    def generate_f(self):
        return [lambda s: s,
                lambda s: self.Round(s)]
    
    def generate_r(self):
        return [lambda s, a: s,
                lambda s, a: a,
                lambda s, a: s,a]
                