
import numpy as np
import copy

class MDP:
    def __init__(self, alpha = 0.0, gamma = 0.0, threshold = 0.000001):
        self.S = [1, 2, 3]
        self.G = [] #goal states
        self.A = ['a', 'b']

        # TODO: action that can be used for grid world
        # self.A = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

        self.P = {} # transition probability
        self.T = {}
        self.V = {}
        self.V_ = {} # last iteration memory for SVI
        self.R = {}
        self.Policy = {} # discrete policy dict

        self.gamma = .9
        self.alpha = .5 # exploration rate used for Q-learning
        self.threshold = 0.000000001 # value iteration threshold

    # =============================================================================================
    # batch init S, G, R, P directly from input
    def set_S(self, in_s=None):  #
        if in_s != None:
            self.S = copy.deepcopy(in_s)
        return 0

    def set_Goal(self, in_g=None):
        if in_g != None:
            self.G = copy.deepcopy(in_g)
        return 0

    def read_R(self, in_r=None):
        if in_r != None:
            self.R = copy.deepcopy(in_r)
        return 0

    def read_P(self, in_p=None):
        if in_p != None:
            self.R = copy.deepcopy(in_p)
        return 0

    # =============================================================================================
    # For reference, vanilla 2D discrete Gridworld, the dynamics is simply s[t+1] = s[t] + a
    # Need to create your own dynamics when applying on continuous world or other specific systems
    def GridDynamics(self, s, a): # specially used in 2D grid world
        return tuple(np.array(s) + np.array(self.A[a]))

    # =============================================================================================
    # TODO: automatic transition probability generation for stochastic 2D discrete system under rule
    # TODO: reference for grid world:
    # TODO: initial value 0, discount gamma = 0.9, P(s,a,s') = p,
    # TODO: P(s,a,random_others) = (1-p) / (#(action)-1)
    # TODO: reward can also be set generally +1 for transition to goal state, and 0 otherwise

    # =============================================================================================
    # every time adding a transition probability, call self.Tsa to store the possible transitions
    # structure of P(s'|s,a): P[prior][posterior], where:
    # prior = (s, a)
    # posterior = s'
    def Psa(self, s, a, s_, p):

        # turn to tuple if the state is an array or list
        [state, state_] = [s, s_] if type(s) != list else [tuple(s), tuple(s_)]

        # self.Tsa(state, a, state_)

        self.add_s(s), self.add_a(a)

        prior, post = (state, a), state_

        if prior not in self.P:
            self.P[prior] = {}

        self.P[prior][post] = p

        return 0

    # =============================================================================================
    # abandoned, not necessary
    # def Tsa(self, s, a, s_):
    #     [state, state_] = [s, s_] if type(s) != list else [tuple(s), tuple(s_)]
    #
    #     index = (state, a)
    #     if index not in self.T:
    #         self.T[index] = []
    #
    #     if state_ not in self.T[index]:
    #         self.T[index].append(state_)
    #
    #     return 0


    # =============================================================================================
    # reward function setting
    # structure of R(s, a): R[index], where:
    # index = (s, a)
    # notice that in some cases, R(s,a,s') also exist, which means the reward is also determined by
    # the next state
    def Rsa(self, s, a, r):
        index = (s, a)
        if index not in self.R:
            self.R[index] = r
        return 0

    # =============================================================================================
    # add single state
    def add_s(self, s):
        if s not in self.S:
            self.S.append(s)
        return 0

    # =============================================================================================
    # add single action
    def add_a(self, a):
        if a not in self.A:
            self.A.append(a)

    # =============================================================================================
    # initialization of value function V
    def init_V(self):
        for state in self.S:
            if state not in self.V:
                self.V[state], self.V_[state] = 0.0, 0.0

    # =============================================================================================
    # turn dict values into numpy array
    def Dict2Vec(self, V):
        v = [V[s] for s in self.S]
        return np.array(v)

    # =============================================================================================
    # summation calculator: sum_j( P(j|s,a) * Vk[j])
    # Sigma(prior) = sum_[post] (self.P[prior][post] * self.V[post])
    # prior = (s, a)
    # post = each element in self.T[prior]
    def Sigma(self, s, a):
        prior = (s, a)
        total = 0.0
        for post in self.P[prior]:
            total += self.P[prior][post] * self.V_[post]
        return total

    # =============================================================================================
    # synchronous value iteration
    # simplified, R(s, a, s') -> R(s, a), and since in grid world the next state is deterministic given (s, a)
    # P(j|s,a) = 1 for some j
    #
    # 1:           Procedure Value_Iteration(S,A,P,R,theta)
    # 2:           Inputs
    # 3:                     S is the set of all states
    # 4:                     A is the set of all actions
    # 5:                     P is state transition function specifying P(s'|s,a)
    # 6:                     R is a reward function R(s,a,s')
    # 7:                     theta a threshold, theta > 0
    # 8:           Output
    # 9:                     pi[S] approximately optimal policy
    # 10:                    V[S] value function
    # 11:          Local
    # 12:                    real array Vk[S] is a sequence of value functions
    # 13:                    action array pi[S]
    #
    # 14:          assign V0[S] arbitrarily
    # 15:          k <- 0
    # 16:          repeat
    # 17:                    k <- k+1
    # 18:                    for each state s do
    # 19:                       Vk[s] = max_a( R(s, a) + lambda * Vk-1[j] )
    # 20:           until for all s satisfies |Vk[s]-Vk-1[s]| < theta
    # 21:           for each state s do
    # 22:                     pi[s] = argmax_a( R(s, a) + lambda * sum_j( P(j|s,a) * Vk[j]) ) )
    # 23:           return pi,Vk
    def SVI(self):
        self.init_V()

        it = 1
        norm_diff = 1

        while norm_diff > self.threshold: # executing value iteration until square_norm(V_current - V_last) <= threshold

            for s in self.S: # TODO: be careful with the type of s here, if list need to modify to tuple
                self.V_[s] = self.V[s]

            for s in self.S:
                max_v = -1.0 * 99999999
                max_a = None
                for a in self.A:
                    v = self.R[s, a] + self.gamma * self.Sigma(s, a)
                    if v > max_v:
                        max_v, max_a = v, a

                self.V[s] = max_v
                self.Policy[s] = max_a

            V_current, V_last = self.Dict2Vec(self.V), self.Dict2Vec(self.V_)

            print V_current, V_last
            it += 1

            norm_diff = np.inner(V_current - V_last, V_current - V_last)

        print it

        return 0

    def VI(self):
        return 0
    def PI(self):
        return 0
    def LP(self):
        return 0
    def DLP(self):
        return 0
    def Q(self):
        return 0

if __name__ == '__main__':
    model = MDP()

    model.Psa(1, 'a', 1, .7)
    model.Psa(1, 'a', 3, .3)
    model.Psa(1, 'b', 2, .6)
    model.Psa(1, 'b', 3, .4)

    model.Rsa(1, 'a', .0)
    model.Rsa(1, 'b', .2)

    model.Psa(2, 'a', 1, .3)
    model.Psa(2, 'a', 3, .7)
    model.Psa(2, 'b', 2, .5)
    model.Psa(2, 'b', 3, .5)

    model.Rsa(2, 'a', .2)
    model.Rsa(2, 'b', 1.)

    model.Psa(3, 'a', 3, 1.)
    model.Psa(3, 'b', 3, 1.)
    model.Rsa(3, 'a', .0)
    model.Rsa(3, 'b', .0)

    model.SVI()
    print model.V, model.V_
    print model.Policy