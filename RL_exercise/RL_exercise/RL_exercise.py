import copy
import pylab
import numpy as np
import tensorflow as tf
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES=2500

class REINFORCEAgent:
    def __init__(self):
        self.load_model=False
        self.action_space=[0,1,2,3,4]
        self.action_size=len(self.action_space)
        self.state_size=15
        self.discount_factor=0.99
        self.learning_rate=0.001

        self.model=self.build_model()
        self.optimizer=self.build_optimizer()
        self.states, self.actions, self.rewards=[],[],[]

        if self.load_model:
            self.model.load_weights('./save_model/reinforce_trained.h5')

    def build_model(self):
        model=Sequential()
        model.add(Dense(24,input_dim=self.state_size,activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        return model

    def build_optimizer(self):
        action=K.placeholder(shape=[None,5])
        discount_rewards=K.placeholder(shape=[None,])

        action_prob=K.sum(action*self.model.output, axis=1)
        cross_entropy=K.log(action_prob)*discount_rewards
        loss=-K.sum(cross_entropy)

        optimizer=Adam(lr=self.learning_rate)
        updates=optimizer.get_updates(self.model.trainable_weights,[], loss)
        train=K.function([self.model.input, action, discount_rewards], [], updates=updates)
        return train
    
    def get_action(self, state):
        policy=self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards):
        discount_rewards=np.zeros_like(rewards)
        running_add=0
        for t in reversed(range(0, len(rewards))):
            running_add=running_add*self.discount_factor+rewards[t]
            discount_rewards[t]=running_add
        return discount_rewards

    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act=np.zeros(self.action_size)
        act[action]=1
        self.actions.append(act)

    def train_model(self):
        discount_rewards=np.float32(self.discount_rewards(self.rewards))
        discount_rewards-=np.mean(discount_rewards)
        discount_rewards/=np.std(discount_rewards)
        self.optimizer([self.states, self.actions, discount_rewards])
        self.states, self.actions, self.rewards=[], [], []

if __name__=="__main__":
    env=Env()
    agent=REINFORCEAgent()

    global_step=0
    scores, episodes=[], []

    for e in range(EPISODES):
        done=False
        score=0
        state=env.reset()
        state=np.reshape(state,[1,15])

        while not done:
            global_step+=1

            action=agent.get_action(state)
            next_state, reward, done=env.step(action)
            next_state=np.reshape(next_state,[1,15])
            agent.append_sample(state, action, reward)

            score+=reward
            state=copy.deepcopy(next_state)

            if done:
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score=round(score,2)
                print("episode:",e, " score:", score," time step:",global_step)

            if e%100==0:
                pylab.plot(episodes, scores,'b')
                pylab.savefig("./save_graph/reinforce.png")
                agent.model.save_weights("./save_model/reinforce.h5")


