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
    
    def rownorm(mat):
    """Row normalization of np.array or pd.DataFrame"""
        return (mat.T / mat.sum(axis=1)).T

    def chiracDelta(goal, vals):
        return [1 if goal == x else 0 for x in column]
    
    def safelog(vals):
        """Silence distracting warnings about log(0)."""
        with np.errstate(divide='ignore'):
            return np.log(vals)    
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
    
    def S_1(self, u, s, a, g, columns):
        # TODO: This should implement equation [8], making sure to normalize over
        # all s,a,g since it's a conditional probability
        lit = self.literal_listener()
        chi = chiracDelta(g, columns)
        utilities = np.exp(self.costs)
        litUtil = [lit*utilities for lit, utilities in zip(lit, utilities)].T
        final = [chi*lit for chi, lit in zip(chi, lit)]
        return rownorm(final)        
        pass
    
    def L_1(self, s, a, u, priorSA, priorG,columns):
        # TODO: This should implement equation [10], making sure to normalize over
        # all u since it's a conditional probability.
        # To iterate over goal functions, use 
        # for g in self.goals().  (each g will be a function)
        priorSAG = priorS*priorG[0] + priorA*priorG[1] + priorSA*priorG[2]
        speak = S_1(self,u,s,a,g,columns)
        final = [priorSAG*speak for priorSAG, speak in zip(priorSAG,speak)]
        np.matmul(priorG,S_1(self,u,s,a,g,columns))
                
    ## must incorporate next 3 functions
    def goals(self):
        return [lambda s, a: r(f(s), a) for f in generate_f for r in generate_r]

    def generate_f(self):
        return [lambda s: s,
                lambda s: self.Round(s)]
    
    def generate_r(self):
        return [lambda s, a: s,
                lambda s, a: a,
                lambda s, a: s,a]
                
        
  if __name__ == '__main__':
    """Examples from the class handout."""

    from IPython.display import display


    def display_reference_game(mod):
        d = mod.lexicon.copy()
        d['costs'] = mod.costs
        d.loc['prior'] = list(mod.prior) + [""]
        d.loc['alpha'] = [mod.alpha] + [" "] * mod.lexicon.shape[1]
        display(d)


    # Core lexicon:

    msgs = s
    s = [30, 32, 1000000]
    a = [0,1]    
    sa = list(zip(s)) + list(zip(a)) + list(itertools.product(s,a)) #each of the possible goals: s,a,s/a
    priorSA = [[1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33], 
               [1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33],
               [1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33]]
    priorS = [sum(x) for x in priorSA]
    priorA = [sum(x) for x in priorSA.T]
    priorG = [1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11]
    lex = pd.DataFrame(
[[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1]], index=msgs, columns=sa) #Edit this function

    print("="*70 + "\nEven priors and all-0 message costs\n")
    basic_mod = RSA(lexicon=lex, prior=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], costs=[0.0, 0.0]) #Edit this function

    display_reference_game(basic_mod)

    print("\nLiteral listener")
    display(basic_mod.literal_listener())

    print("\nPragmatic speaker")
    display(basic_mod.speaker())

    print("\nPragmatic listener")
    display(basic_mod.listener())


    print("="*70 + "\nEven priors, imbalanced message costs\n")
    cost_most = RSA(lexicon=lex, prior=[0.5, 0.5], costs=[-6.0, 0.0])

    display_reference_game(cost_most)
