from maze_env import Maze
from RL_brain import Agent

def update():
    #episode代表回合，每玩一次游戏都是一个回合
    for episode in range(100):
        observation=env.reset()  #重置并得到初始状态
        while True:
            #界面更新
            env.render()
            action=agent.choose_action(str(observation))
            #得到对应action后的状态 奖励 是否结束
            observation_, reward,done=env.step(action)
            #更新上一个状态到现在这个状态对应行为的Q值
            agent.learn(str(observation),action,reward,str(observation_))
            #下一次的观察值是当前的观察值
            observation=observation_
            if done: #碰到障碍物或者目的地都会游戏结束终止，但奖励不一样
                break
    print('game over')
    env.destroy()

if __name__=='__main__':
    env=Maze()
    #range(env.n_actions)返回一个列表
    agent=Agent(actions=list(range(env.n_actions)))
    env.after(100,update)
    env.mainloop()
