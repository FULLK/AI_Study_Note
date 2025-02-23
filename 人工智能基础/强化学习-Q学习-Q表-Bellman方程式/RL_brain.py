import numpy as np
import pandas as pd

class Agent:
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions=actions
        #self.q_table 是一个 Pandas DataFrame，其 行 表示不同的状态（状态 state），而 列 表示不同的行为（action）。
        self.q_table=pd.DataFrame(columns=self.actions,dtype=float)
        self.epsilon=e_greedy
        self.gamma=reward_decay
        self.lr=learning_rate

    #ε-贪婪策略 (ε-greedy strategy)
    # 即在多数情况下选择当前认为最优的动作，但也会有一定的概率选择一个随机动作，从而保持探索
    def choose_action(self, observation):#observation 是当前环境的状态（state）
        self.check_state_exists(observation)
        #我们通常选择 Q值最大的动作（即当前认为最优的行为），但为了避免陷入局部最优解，偶尔也要 随机探索（即尝试其他动作）。
        if np.random.uniform()<self.epsilon:
            #生成一个在 [0, 1) 区间内的随机浮点数。如果该值小于 self.epsilon 就进行 贪婪选择，否则进行随机选择。
            #loc[行标签, 列标签]
            state_actions=self.q_table.loc[observation,:]
            #据布尔条件筛选出 state_actions 中所有 Q值等于最大 Q值 的动作及其对应的 Q值。然后返回其索引，有多个同样大的就随机取一个
            action = np.random.choice(state_actions[state_actions==np.max(state_actions)].index)
        else:
            action=np.random.choice(self.actions)    
        return action
 
    def learn(self,s,a,r,s_):
        ## Q(s,a) <- Q(s,a) + lr * [r + γ*max(Q(s_,a)) - Q(s,a)]
        # s, a, r, s_ 分别对应的就是
        # 当前时刻的state状态，当前时刻的action行为，
        # 环境接收到当前时刻的action之后给出来的reward奖励，下一个时刻的state状态
        self.check_state_exists(s_)
        q_predict=self.q_table.loc[s,a]
        if s_!='terminal':
            q_target=r+self.gamma*self.q_table.loc[s_,:].max() #r + γ*max(Q(s_,a))
        else:
            q_target=r 
        self.q_table.loc[s,a]+=self.lr*(q_target-q_predict)

    def check_state_exists(self, state):
        #表示检查给定的 state 是否已经存在于 Q-table 中的行索引里
        if state not in self.q_table.index:
            # self.q_table=self.q_table.append(
            #     #pd.Series 用来创建一个 Series 对象，它是 DataFrame 中的一行数据。创建时需要指定数据、索引和名称。
            #     pd.Series(
            #         [0]*len(self.actions),
            #         index=self.q_table.columns,
            #         name=state
            #     )
            # )
              # 创建新的一行
            new_row = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            # 使用 pd.concat 合并到 q_table
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])


