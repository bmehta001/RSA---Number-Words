""" Bhagirath Mehta and Suvir Mirchandani
    Final Project: Implementing an Extension of the Rational Speech Acts Model for the
    Figurative Use of Number Words
    Ling 130a/230a: Introduction to Semantics and Pragmatics, Winter 2019
"""

import itertools
import math
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Bhagirath Mehta, Suvir Mirchandani"

class NonliteralNumbersRSA:
    """Implementation of the Rational Speech Acts model extended for nonliteral number words.

        Based on an RSA implementation by Prof. Chris Potts
        (http://web.stanford.edu/class/linguist130a/materials/rsa130a.py)

        and the following work:

        Kao, Justine T., Jean Y. Wu, Leon Bergen & Noah D. Goodman. 2014. Nonliteral
        understanding of number words. Proceedings of the National Academy of Sciences,
        111(33). 12002-12007. doi: 10.1073/pnas.1407479111.
    
    Parameters
    ----------
    lexicon : list
        possible utterances
    states : list
        possible states; should be equal to lexicon based on Kao et al. (2014)
    affects : list
        possible affect choices; in this case, binary
    s_prior: list
        prior probabilities for the states
    sa_prior: list of lists
        prior probabilities for the affects given certain state;
        example: sa_prior[0][1] is the probability of having affect=1 for the state at
        index 0
    round_cost: float
        cost of uttering a round number
    sharp_cost: float
        cost of uttering a non-round number
    precision: int
        rounding place value; rounding happens to the nearest 10^precision
        example: precision=1 considers 12 to be 10 when rounded
    """
    def __init__(self, lexicon, states, affects, s_prior, sa_prior, round_cost, sharp_cost,
        precision):
        assert lexicon == states, "lexicon (U) must be equal to states (S)"
        self.lexicon = lexicon
        self.states = states
        self.affects = affects
        # A list of tuples for all state/affect combinations
        self.meanings = list(itertools.product(self.states, self.affects)) 
        self.s_prior = s_prior
        self.sa_prior = sa_prior
        self.round_cost = round_cost
        self.sharp_cost = sharp_cost
        self.precision = precision

    def literal_listener(self):
        """ Literal listener predictions for all possible states and affects given an
        utterance.

        Returns
        -------
        3D list.  The first dimension corresponds to utterances, the second to states, and
        the third to affects.
        """
        literal = [[[0 for a in self.affects] for s in self.states] for u in self.lexicon]

        for i, u in enumerate(self.lexicon):
            for j, s in enumerate(self.states):
                for k, a in enumerate(self.affects):
                    literal[i][j][k] = self.L_0(s, a, u)
        return literal
    
    def pragmatic_speaker(self):
        """ Pragmatic predictions for all possible utterances given states, affects, and goals.

        Returns
        -------
        4D list.  The first dimension corresponds to goals, the second to affects, the
        third to states, and the fourth to utterances.
        """
        speaker = [[[[0 for u in self.lexicon] for s in self.states] for a in self.affects]
            for g in self.goals()]

        for i, g in enumerate(self.goals()):
            for j, a in enumerate(self.affects):
                for k, s in enumerate(self.states):
                    for l, u in enumerate(self.lexicon):
                        speaker[i][j][k][l] = self.S_1(u, s, a, g)
        return speaker
    
    def pragmatic_listener(self):
        """ Pragmatic listener predictions for all possible states and affects given an
        utterance.

        Returns
        -------
        3D list.  The first dimension corresponds to utterances, the second to states, and
        the third to affects.
        """
        listener = [[[0 for a in self.affects] for s in self.states] for u in self.lexicon]
        
        for i, u in enumerate(self.lexicon):
            for j, s in enumerate(self.states):
                for k, a in enumerate(self.affects):
                    listener[i][j][k] = self.L_1(s, a, u)
        return listener
    
    def P_A(self, a, s):
        """Returns the prior probability that a certain affect `a` is true given that a
        specific state is true"""
        return self.sa_prior[self.states.index(s)][a]
    
    def P_S(self, s):
        """Returns the prior probability that a certain state `s` is true"""
        return self.s_prior[self.states.index(s)]
    
    def Round(self, x):
        """Returns a rounded version of utterance, based on `self.precision`"""
        return round(x, -self.precision)
    
    def C(self, u):
        """Returns the cost of an utterance based on whether it is round or sharp"""
        if u == self.Round(u):
            return self.round_cost
        return self.sharp_cost
    
    def L_0(self, s, a, u):
        """Returns the literal listener's prediction probability of a state `s` and affect
        `a` given an utterance `u`"""
        if (s == u):
            return self.P_A(a,s)
        return 0
    
    def L_0_projected(self, x, u, g):
        """Version of `L_O` projected onto a specific goal `x` (unused in our implementation)"""
        prob = 0
        for s_p, a_p in self.meanings:
            if x == g(s_p, a_p):
                prob += self.L_0(s_p, a_p, u)
        return prob


    def S_1(self, u, s, a, g):
        """Returns the speaker's prediction probability of an utterance `u` given a state `s`,
        affect `a`, and goal `g`.  Uses helper function `S_1_joint` for joint probability."""
        numerator = self.S_1_joint(u, s, a, g)
        # Normalization
        denominator = sum([self.S_1_joint(u_p, s, a, g) for u_p in self.lexicon])
        return numerator / denominator

    def S_1_joint(self, u, s, a, g):
        """Returns the speaker's joint probability of an utterance `u`, state `s`, affect `a`,
        and goal `g`"""
        total = 0
        for s_p, a_p in self.meanings:
            if g(s, a) == g(s_p, a_p):
                total += self.L_0(s_p, a_p, u) * math.exp(-1 * self.C(u))
        return total

    def L_1(self, s, a, u):
        """Returns the pragmatic listener's prediction probability of a state `s` and affect
        `a` given an utterance `u`.  Uses helper function `L_1_joint` for joint probability."""

        numerator = self.L_1_joint(s, a, u)
        # Normalization
        denominator = sum([self.L_1_joint(s_p, a_p, u) for s_p in self.states for a_p in
            self.affects])
        return numerator / denominator

    def L_1_joint(self, s, a, u):
        """Returns the pragmatic speaker's joint probabiltiy of a state `s`, affect `a`, and
        utterance `u`."""
        total = 0
        for g in self.goals():
            total += self.P_S(s) * self.P_A(a, s) * self.P_G(g) * self.S_1(u, s, a, g)
        return total

    def P_G(self, g):
        """Returns the probability of a particular conversational goal.  Based on Kao et al. (2014),
        we implemetn this as a uniform prior."""
        return 1. / len(self.goals())

    def goals(self):
        """Returns a list corresponding to the different conversational goals described by Kao et
        al. (2014).  These are functions that return (state), (affect), or (state, affect) for
        either exact or approximated states, for a total of six goal functions."""
        return [lambda s, a, r=r, f=f: r(f(s), a) for r in self.possible_r() for f in
            self.possible_f()]

    def possible_f(self):
        """Returns a list of functions `f` of state that return state or rounded state."""
        return [lambda s: s,
                lambda s: self.Round(s)]
    
    def possible_r(self):
        """Returns a list of functions `r` that return (state), (affect), or (state, affect)."""
        return [lambda s, a: (s,),
                lambda s, a: (a,),
                lambda s, a: (s,a)]

    def display_listener(self, listener, title, visual):
        """Displays the probability distribution for a listener (either `self.literal_listener()
        or self.pragmatic_listener()).  If visual=True, displays graphically with a heatmap-like
        representation for the probabilities."""
        
        print("="*70 + "\n" + title + ": ")
        
        if visual:
            sns.set()
            f, axes = plt.subplots(1, len(self.lexicon), figsize=(15,5))
            f.suptitle(title, fontsize=16)
            f.subplots_adjust(wspace=0.5, top=0.85, bottom=0.15)
            cbar_ax = f.add_axes([.93, .15, .03, .7])

        for u, given_u in enumerate(listener):
            lex = pd.DataFrame(index = self.states, columns = self.affects, data = given_u)
            d = lex.copy()
            if visual:
                fig = sns.heatmap(d, annot=True, fmt = '.2g', ax = axes[u], vmin=0, vmax=1,
                    linewidths=2, cmap="Blues", cbar=(u == len(listener) - 1),
                    cbar_ax=cbar_ax if (u == len(listener) - 1) else None)
                fig.set_xlabel("Affects")
                fig.set_ylabel("States")
                fig.set_title("Utterance: " + str(self.lexicon[u]))
                fig.tick_params(axis='x',bottom=False, top=False)
                fig.tick_params(axis='y',left=False, right=False)
            else:
                d.loc['utterance'] = [self.lexicon[u]] + [" "]
                display(d)
                print

        if visual:
            plt.show()

    def display_speaker(self, speaker, title, visual):
        """Displays the probability distribution for a speaker (self.pragmatic_speakers()).
        If visual=True, displays graphically with a heatmap-like representation for the
        probabilities."""

        print("="*70 + "\n" + title + ": ")
        goals = ["r_{}(f_{}(s),a)".format(r,f) for r in ['s','a','sa'] for f in ['e','a']]

        if visual:
            sns.set()
            f, axes = plt.subplots(len(goals), len(self.affects), figsize=(6,12))
            f.suptitle(title, fontsize=16)
            f.subplots_adjust(wspace=0.3, hspace=0.8, top=0.92, bottom=0.08)
            cbar_ax = f.add_axes([.93, .15, .02, .7])

        for g, given_g in enumerate(speaker):
            for a, given_ag in enumerate(given_g):
                lex = pd.DataFrame(index = self.states, columns = self.lexicon, data = given_ag)
                d = lex.copy()
                if visual:
                    fig = sns.heatmap(d, annot=True, fmt = '.2g', ax = axes[g][a], vmin=0, vmax=1,
                        linewidths=2, cmap="Reds", cbar=(g == len(goals) - 1 and
                        a == len(self.affects) - 1), cbar_ax=cbar_ax if (g == len(goals) - 1
                        and a == len(self.affects) - 1) else None)
                    fig.set_xlabel("Utterances")
                    fig.set_ylabel("States")
                    fig.set_title("Goal: " + str(goals[g]) + "; Affect: " + str(self.affects[a]))
                    fig.tick_params(axis='x',bottom=False, top=False)
                    fig.tick_params(axis='y',left=False, right=False)
                else:
                    d.loc['goal'] = [goals[g]] + [" "] * 2
                    d.loc['affect:'] = [rsa.affects[a]] + [" "] * 2
                    display(d)
                    print

        if visual:
            plt.show()
        

if __name__ == '__main__':

    # Core lexicon:
    S = U = [30, 32, 1000]
    A = [0, 1]
    s_prior = [0.495, 0.495, 0.01]
    sa_prior = [ [0.9, 0.1],
                 [0.9, 0.1], 
                 [0.01, 0.99] ] # in the form sa_prior[s][a] = P(a|s)

    # The parameters of the model can be changed here.
    rsa = NonliteralNumbersRSA(lexicon=U, states=S, affects=A, s_prior=s_prior,
                               sa_prior=sa_prior, round_cost=1, sharp_cost=5, precision=1)

    rsa.display_listener(rsa.literal_listener(), title="Literal Listener", visual=True)
    
    rsa.display_speaker(rsa.pragmatic_speaker(), title="Pragmatic Speaker", visual=True)

    rsa.display_listener(rsa.pragmatic_listener(), title="Pragmatic Listener", visual=True)
