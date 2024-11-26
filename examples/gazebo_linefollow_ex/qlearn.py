import random
import pickle

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        # Load stored values
        with open(filename, 'rb') as f:
            self.q = pickle.load(f)
        print("Loaded file: {}".format(filename))



    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file and CSV file.
        '''
        # Save stored values in a pickle file
        with open(filename, 'wb') as f:
            pickle.dump(self.q, f)
        print("Wrote to file: {}".format(filename))
        



    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward + gamma * max(Q(s')) - Q(s, a))
        '''
        oldv = self.q.get((state, action), 0.0)
        self.q[(state, action)] = oldv + self.alpha * (value - oldv)
        # Ensure Q-values do not become negative
        self.q[(state, action)] = max(0, self.q[(state, action)])
        print(f"Updated Q-value for state {state}, action {action}: {self.q[(state, action)]}")

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            # Explore: choose a random action
            action = random.choice(self.actions)
        else:
            # Exploit: choose the action with the highest Q-value
            count = q.count(maxQ)
            # In case there are several state-action max values, we select a random one among them
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            action = self.actions[i]

        if return_q:  # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        value = reward + self.gamma * maxqnew
        self.learnQ(state1, action1, value, value)

        # Maintain a higher exploration rate for a longer period to ensure sufficient exploration
        self.epsilon = max(0.1, self.epsilon * 0.9995)