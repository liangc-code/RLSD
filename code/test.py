from MyFCN import *
import sys
import time
import State_mix
from pixelwise_a3c import *
import numpy as np
import scipy.io as scio
import pyortho as lo # obtained from (https://github.com/chenyk1990/pyortho).

#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE = 0.0001
TEST_BATCH_SIZE = 1
EPISODE_LEN = 4 + 1
GAMMA = 0.95

N_ACTIONS =19
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

def test(agent, fout):
    sum_reward = 0
    sum_reward_trans = 0
    current_state = State_mix.State_mix((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    raw_x = scio.loadmat('../data/test/clean.mat')['clean'].astype(np.float32)
    raw_n = scio.loadmat('../data/test/input.mat')['input'].astype(np.float32)
    raw_x = raw_x.reshape(1, 1, raw_x.shape[0], raw_x.shape[1]).astype(np.float32)
    raw_n = raw_n.reshape(1, 1, raw_n.shape[0], raw_n.shape[1]).astype(np.float32)
    raw_n = (raw_n) / np.max(abs(raw_n))
    current_state.reset(raw_x, raw_n)

    for t in range(0, EPISODE_LEN):
        previous_image = current_state.image.copy()
        action, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)
        reward_mse = np.square(raw_x - previous_image) - np.square(raw_x - current_state.image)
        reward = reward_mse - 0.1 * ract(action) / 3
        sum_reward += np.mean(reward)*np.power(GAMMA,t)

    # fine-tuning
    current_state_train = current_state
    allreward = []
    episodes = 1
    for episode in range(0, episodes):
        t = 0
        previous_image = current_state_train.image.copy()
        action, inner_state = agent.act_and_train(current_state_train.tensor, reward)
        current_state_train.step(action, inner_state)
        simip = lo.localsimi(np.squeeze(previous_image), np.squeeze(raw_x - previous_image), [5, 5, 1], niter=20, eps=0, verb=1)
        simic = lo.localsimi(np.squeeze(current_state_train.image), np.squeeze(raw_x - current_state_train.image), [5, 5, 1], niter=20, eps=0, verb=1)
        reward_simi = simip - simic
        reward_simi = reward_simi.reshape(reward_simi.shape[2], 1, reward_simi.shape[0], reward_simi.shape[1]).astype(np.float32)
        reward = reward_simi
        sum_reward_trans += np.mean(reward) * np.power(GAMMA, t)
        allreward.append(sum_reward_trans)
        agent.stop_episode_and_train(current_state_train.tensor, reward, True)
        agent.optimizer.alpha = LEARNING_RATE * ((1 - episode / episodes) ** 0.9)
    sys.stdout.flush()

    for t in range(EPISODE_LEN, EPISODE_LEN + 1):
        previous_image = current_state.image.copy()
        action, inner_state = agent.act(current_state.tensor)
        current_state.step(action, inner_state)
        result = current_state.image
        reward_mse = np.square(raw_x - previous_image) - np.square(raw_x - current_state.image)
        reward = reward_mse - 0.1 * ract(action) / 3
        sum_reward += np.mean(reward)*np.power(GAMMA,t)

    agent.stop_episode()
 
    print("test total reward {a}".format(a=sum_reward))
    fout.write("test total reward {a}\n".format(a=sum_reward))
    sys.stdout.flush()

def main(fout):
    chainer.cuda.get_device_from_id(GPU_ID).use()
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    model.conv1.W.update_rule.enabled = False
    model.conv1.b.update_rule.enabled = False
    model.diconv2.diconv.W.update_rule.enabled = False
    model.diconv2.diconv.b.update_rule.enabled = False
    model.diconv3.diconv.W.update_rule.enabled = False
    model.diconv3.diconv.b.update_rule.enabled = False
    model.diconv4.diconv.W.update_rule.enabled = False
    model.diconv4.diconv.b.update_rule.enabled = False
    model.diconv5_pi.diconv.W.update_rule.enabled = False
    model.diconv5_pi.diconv.b.update_rule.enabled = False
    model.diconv6_pi.diconv.W.update_rule.enabled = False
    model.diconv6_pi.diconv.b.update_rule.enabled = False
    model.conv7_Wz.W.update_rule.enabled = False
    model.conv7_Uz.W.update_rule.enabled = False
    model.conv7_Wr.W.update_rule.enabled = False
    model.conv7_Ur.W.update_rule.enabled = False
    model.conv7_W.W.update_rule.enabled = False
    model.conv7_U.W.update_rule.enabled = False
    model.conv8_pi.model.W.update_rule.enabled = True
    model.conv8_pi.model.b.update_rule.enabled = True

    model.diconv5_V.diconv.W.update_rule.enabled = False
    model.diconv5_V.diconv.b.update_rule.enabled = False
    model.diconv6_V.diconv.W.update_rule.enabled = False
    model.diconv6_V.diconv.b.update_rule.enabled = False
    model.conv7_V.W.update_rule.enabled = True
    model.conv7_V.b.update_rule.enabled = True

    model.conv_R.W.update_rule.enabled = False

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('./model/RLSD_30000/model.npz', agent.model)
    chainer.serializers.load_npz('./model/RLSD_30000/model.npz', agent.optimizer.target)
    agent.act_deterministically = True
    agent.model.to_gpu()

    #_/_/_/ testing _/_/_/
    test(agent, fout)

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
