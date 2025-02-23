import game.wrapped_flappy_bird as game
from utils import resize_and_bgr2gray, image_to_tensor
from agent import DeepQNetwork
import argparse
import torch.nn as nn
import torch
from random import random, randint, sample
import numpy as np

def get_args():
    parser=argparse.ArgumentParser("训练深度神经网络Q学习去玩FlappyBird")
    parser.add_argument("--image_size",type=int,default=84,help="代表图像的高和宽")
    parser.add_argument("--batch_size",type=int,default=32,help="表示训练时使用的批处理大小")
    parser.add_argument("--optimizer",type=str,choices=["sgd","adam"],default="adam",help="指定优化器的类型。可选值是 sgd（随机梯度下降）和 adam（一种自适应学习率的优化算法）")
    parser.add_argument("--lr",type=float,default=1e-6,help="表示学习率，类型为浮动数，默认值是 1e-6")
    parser.add_argument("--gamma",type=float,default=0.99,help="表示折扣因子（discount factor）。它控制了模型在Q学习中对未来奖励的重视程度。值越接近1，模型会更加重视未来的奖励。默认值是 0.99")
    parser.add_argument("--initial_epsilon",type=float,default=0.1,help="表示ε-贪婪策略中的初始ε值。epsilon 控制着探索与利用的平衡，值越大模型越倾向于探索（即随机选择动作），值越小则越倾向于利用当前策略")
    parser.add_argument("--final_epsilon",type=float,default=1e-4,help="表示ε-贪婪策略中的最终ε值")
    parser.add_argument("--num_iters",type=int,default=2000000,help="表示训练的迭代次数")
    parser.add_argument("--saved_path",type=str,default="trained_models",help="指定训练完的模型存储路径")
    parser.add_argument("--history_memory_size",type=int,default=50000,help="存储模型历史交互信息以供后续训练")
    args = parser.parse_args()
    return args


def train(args):
    #初始化
    #如果GPU可用，设置随机种子
    #否则，设置CPU的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    #获取模型和设置优化器和损失函数
    model=DeepQNetwork()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion=nn.MSELoss()
    #初始化游戏环境和初始行为
    game_state=game.GameState()
    init_action=torch.zeros(2,dtype=torch.float32)
    init_action[1]=1
    #通过初始行为获得初始图片数据、奖励和终止状态
    init_image_data, init_reward, init_terminal=game_state.frame_step(init_action)
    init_image_data=resize_and_bgr2gray(init_image_data)
    init_image_data=image_to_tensor(init_image_data)
    #四张图片组合成状态
    state=torch.cat((init_image_data,init_image_data,init_image_data,init_image_data)).unsqueeze(0)
    #历史交互信息
    history_memory=[]
    #迭代次数
    iter=0

    while iter<args.num_iters:
        ##一行两列 对应两个动作的Q值或者说概率
        predicted_action=model(state)[0] 
        #根据epsilon和Q值来选择要为1的动作位置
        #随着迭代次数增加，eplison逐渐减小，随机数小于eplison的概率也变小，逐渐倾向选择Q值大的行为
        epsilon=args.final_epsilon+ ( (args.num_iters-iter)*(args.initial_epsilon-args.final_epsilon) / args.num_iters )
        u = random()
        random_action=u<=epsilon
        if random_action:
            print("采取随机行为")
            action_index=randint(0,1)
        else:
            print("采取Q值最大的行为")
            action_index=torch.argmax(predicted_action).item()
        #将要为1的位置设置为1即设置好了动作
        action=torch.zeros(2,dtype=torch.float32)
        action[action_index]=1
        #执行动作，获得图片数据、奖励和终止状态
        image_data, reward, terminal=game_state.frame_step(action)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        print(state.size(),image_data.size())
        #更新下一个时刻状态
        next_state=torch.cat( (state.squeeze(0)[1:,:,:] ,image_data) ).unsqueeze(0)
        #将交互信息存储到历史交互信息中
        history_memory.append([state,action,reward,next_state,terminal])
        if len(history_memory)>args.history_memory_size:
            del history_memory[0]
        #选择一个batch的信息来训练模型
        batch=sample(history_memory,min(len(history_memory) ,args.batch_size ))
        print(len(batch))
        #zip可以理解为每一列变成一个元组，然后对其中的每一个元组进行转换为tensor张量
   
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch=zip(*batch)
        state_batch=torch.cat(tuple(state for state in state_batch))

        action_batch=torch.cat(tuple(action for action in action_batch ))
        reward_batch=torch.from_numpy(np.array(reward_batch,dtype=np.float32))[:,None]
        next_state_batch=torch.cat(tuple(next_state for next_state in next_state_batch ))
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        #得到一个batch的当前预测和下一个预测
        print(state_batch.size(),next_state_batch.size())
        current_prediction_batch=model(state_batch)
        next_prediction_batch=model(next_state_batch)

         # 准备y target对于一个批次的
        # 没有terminal就是target = R+gamma*Max(Q(S',a))
        # 有terminal就是target = R
        y_batch = torch.cat(tuple(
            reward if terminal else reward + args.gamma * torch.max(prediction) for reward, terminal, prediction in
            zip(reward_batch, terminal_batch, next_prediction_batch
                )))
        #就是每次结果中选择的那个行为的概率就是Q值
        q_value = torch.sum(current_prediction_batch*action_batch.view(-1, 2), dim=1)
        optimizer.zero_grad()
        loss = criterion(q_value, y_batch) #计算损失
        loss.backward()
        optimizer.step()
        state=next_state
        iter += 1

        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon: {}, Reward: {}, Q-value: {}".format(
            iter+1, args.num_iters, action, loss, epsilon, reward, torch.max(predicted_action)))
        # 间隔一些迭代，保存模型
        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(args.saved_path, iter+1))
    torch.save(model, "{}/flappy_bird".format(args.saved_path))

if __name__ == '__main__':
   args=get_args()
   train(args)