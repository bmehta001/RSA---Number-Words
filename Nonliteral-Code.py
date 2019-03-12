# Note: this code is a work in progress.  It needs to be cleaned up, but at this point,
# we believe it accurately reproduces the model of Kao et al. and generalizes it a bit.
# We will test this more thoroughly.  We also hope to create some sort of visual
# of the output to put in the paper.

import itertools
import math

class NonliteralNumbersRSA:
    
    # TODO: Explain parameters
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

    def literal_listener(self):
        # Table of L_0 for all u,s,a
        literal = [[[0 for a in self.affects] for s in self.states] for u in self.lexicon]

        for i, u in enumerate(self.lexicon):
            for j, s in enumerate(self.states):
                for k, a in enumerate(self.affects):
                    literal[i][j][k] = self.L_0(s, a, u)
        return literal
    
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
    
    def listener(self):
        # Table of L_1 for all u, s, a
        listener = [[[0 for a in self.affects] for s in self.states] for u in self.lexicon]
        
        for i, u in enumerate(self.lexicon):
            for j, s in enumerate(self.states):
                for k, a in enumerate(self.affects):
                    listener[i][j][k] = self.L_1(s, a, u)
        return listener
    
    def P_A(self, a, s):
        return self.sa_prior[self.states.index(s)][a]
    
    def P_S(self, s):
        return self.s_prior[self.states.index(s)]
    
    def Round(self, x):
        return round(x, -self.precision)
    
    def C(self, u):
        if u == self.Round(u):
            return self.round_cost
        return self.sharp_cost
    
    def L_0(self, s, a, u):
        if (s == u):
            return self.P_A(a,s)
        return 0
    
    def L_0_projected(self, x, u, g):
        prob = 0
        for s_p, a_p in self.meanings:
            if x == g(s_p, a_p):
                prob += self.L_0(s_p, a_p, u)
        return prob

    def S_1(self, u, s, a, g):
        numerator = self.S_1_helper(u, s, a, g)
        denominator = sum([self.S_1_helper(u_p, s, a, g) for u_p in self.lexicon])
        return numerator / denominator

    def S_1_helper(self, u, s, a, g):
        total = 0
        for s_p, a_p in self.meanings:
            if g(s, a) == g(s_p, a_p):
                total += self.L_0(s_p, a_p, u) * math.exp(-1 * self.C(u))
        return total

    def L_1(self, s, a, u):
        numerator = self.L_1_helper(s, a, u)
        denominator = sum([self.L_1_helper(s_p, a_p, u) for s_p in self.states for a_p in
            self.affects])
        return numerator / denominator

    def L_1_helper(self, s, a, u):
        total = 0
        for g in self.goals():
            total += self.P_S(s) * self.P_A(a, s) * self.P_G(g) * self.S_1(u, s, a, g)
        return total

    def P_G(self, g):
        return 1. / len(self.goals())

    def goals(self):
        return [lambda s, a, r=r, f=f: r(f(s), a) for r in self.generate_r() for f in
            self.generate_f()]

    def generate_f(self):
        return [lambda s: s,
                lambda s: self.Round(s)]
    
    def generate_r(self):
        return [lambda s, a: (s,),
                lambda s, a: (a,),
                lambda s, a: (s,a)]
                
        
if __name__ == '__main__':
    """Examples from the class handout."""

    from IPython.display import display


#     def display_reference_game(mod):
#         d = mod.lexicon.copy()
#         d['costs'] = mod.costs
#         d.loc['prior'] = list(mod.prior) + [""]
#         d.loc['alpha'] = [mod.alpha] + [" "] * mod.lexicon.shape[1]
#         display(d)


    # Core lexicon:

    S = U = [30, 32, 1000000]
    A = [0, 1]
    s_prior = [0.55, 0.44, 0.01]
    sa_prior = [ [0.9, 0.1],
                 [0.9, 0.1], 
                 [0.1, 0.9] ]

    rsa = NonliteralNumbersRSA(lexicon=U, states=S, affects=A, s_prior=s_prior,
                               sa_prior=sa_prior, round_cost=1, sharp_cost=3, precision=1)

    literal_listener = rsa.literal_listener()

    for u, given_u in enumerate(literal_listener):
        print "utterance:", rsa.lexicon[u]
        for s, for_s_given_u in enumerate(given_u):
            print "\tstate:", rsa.states[s]
            for a, for_sa_given_u in enumerate(for_s_given_u):
                print "\t\taffect:", rsa.affects[a]
                print "\t\t\tP(s,a|u):", for_sa_given_u

    speaker = rsa.speaker()

    goals = ["r_{}(f_{}(s),a)".format(r,f) for r in ['s','a','sa'] for f in ['e','a']]
    for g, given_g in enumerate(speaker):
        print "goal", goals[g]
        for s, given_sg in enumerate(given_g):
            print "\tstate:", rsa.states[s]
            for a, given_sag in enumerate(given_sg):
                print "\t\taffect:", rsa.affects[a]
                for u, for_u_given_sag in enumerate(given_sag):
                    print "\t\t\tutterance:", rsa.lexicon[u]
                    print "\t\t\t\tP(u|s,a,g)", for_u_given_sag

    listener = rsa.listener()

    for u, given_u in enumerate(listener):
        print "utterance:", rsa.lexicon[u]
        for s, for_s_given_u in enumerate(given_u):
            print "\tstate:", rsa.states[s]
            for a, for_sa_given_u in enumerate(for_s_given_u):
                print "\t\taffect:", rsa.affects[a]
                print "\t\t\tP(s,a|u):", for_sa_given_u

#     a = [0,1]    
#     sa = list(zip(s)) + list(zip(a)) + list(itertools.product(s,a)) #each of the possible goals: s,a,s/a
#     priorSA = [[1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33], 
#                [1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33],
#                [1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33, 1/33]]
#     priorS = [sum(x) for x in priorSA]
#     priorA = [sum(x) for x in priorSA.T]
#     priorG = [1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11, 1/11]
#     lex = pd.DataFrame(
# [[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1]], index=msgs, columns=sa) #Edit this function

#     print("="*70 + "\nEven priors and all-0 message costs\n")
#     basic_mod = RSA(lexicon=lex, prior=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6], costs=[0.0, 0.0]) #Edit this function

#     display_reference_game(basic_mod)

#     print("\nLiteral listener")
#     display(basic_mod.literal_listener())

#     print("\nPragmatic speaker")
#     display(basic_mod.speaker())

#     print("\nPragmatic listener")
#     display(basic_mod.listener())


#     print("="*70 + "\nEven priors, imbalanced message costs\n")
#     cost_most = RSA(lexicon=lex, prior=[0.5, 0.5], costs=[-6.0, 0.0])

#     display_reference_game(cost_most)
