from MyFCN import *
import sys
import time
import State_mix
from pixelwise_a3c import *
import numpy as np
import scipy.io as scio

#_/_/_/ paths _/_/_/
SAVE_PATH = "./model/RLSD_"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1
N_EPISODES = 30000
EPISODE_LEN = 5
SNAPSHOT_EPISODES = 3000
TEST_EPISODES = 3000
GAMMA = 0.95

N_ACTIONS = 19
MOVE_RANGE = 3
CROP_SIZE = 70

GPU_ID = 0

def ract(act):
    b, h, w = act.shape
    actt0 = np.hstack((act[:, 0:1, :], act[:, 0:h - 1, :]))
    actt2 = np.hstack((act[:, 1:h, :], act[:, h - 1:h, :]))
    actt = 3 - (actt0 == act) - (actt0 == actt2) - (act == actt2)
    actt = np.where(actt == 0, 1, actt)
    actt = actt.reshape(b, 1, h, w).astype('float32')
    return actt

def main(fout):
    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State_mix.State_mix((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    agent.model.to_gpu()
    
    #_/_/_/ training _/_/_/
    traindata = scio.loadmat('../data/train/train_dataset.mat')['im'].astype(np.float32)
    train_data_size = traindata.shape[2]
    indices = np.random.permutation(train_data_size)
    i = 0
    allreward = []
    for episode in range(1, N_EPISODES+1):
        # display current state
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        # load data
        r = indices[i:i+TRAIN_BATCH_SIZE]
        mini_batch_size = len(r)
        in_channels = 1
        raw_x = np.zeros((mini_batch_size, in_channels, CROP_SIZE, CROP_SIZE)).astype(np.float32)
        raw_n = np.zeros((mini_batch_size, in_channels, CROP_SIZE, CROP_SIZE)).astype(np.float32)
        for ii, index in enumerate(r):
            raw_x[ii, 0, :, :] = (traindata[:, :, index, 0]).astype(np.float32)
            raw_n[ii, 0, :, :] = (traindata[:, :, index, 1]).astype(np.float32)
        # initialize the current state and reward
        current_state.reset(raw_x,raw_n)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            reward_mse = np.square(raw_x - previous_image) - np.square(raw_x - current_state.image)
            reward = reward_mse - 0.1*ract(action)/3
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        allreward.append(sum_reward)
        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {a}".format(a=sum_reward))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))
        
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)

if __name__ == '__main__':
        fout = open('log.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
