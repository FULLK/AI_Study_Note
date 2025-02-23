import numpy as np
import time
import sys
print(sys.version_info.major)
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels 定义每个格子的边长为 40 像素。
MAZE_H = 4  # grid height 高度
MAZE_W = 4  # grid width 宽度

#类继承了 tk.Tk，意味着它是一个 Tkinter 窗口应用程序
class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        #action_space 包含四个动作：上 ('u')、下 ('d')、左 ('l') 和右 ('r')。
        self.action_space = ['u', 'd', 'l', 'r']
        #n_actions 是动作空间的大小，即智能体可以采取的不同动作的数量。
        self.n_actions = len(self.action_space)
        self.title('maze')
        #self.geometry 设置了迷宫的尺寸。
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        #创建一个 Canvas 画布，用于绘制迷宫，背景颜色为白色，迷宫大小由 MAZE_H 和 MAZE_W 控制，每个格子的尺寸是 UNIT。
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        #在画布上画出纵横的网格线，每隔 UNIT 像素画一条水平和垂直的线，形成一个 MAZE_H × MAZE_W 的网格。
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin  X坐标和y坐标 代表最左上角的格子的中心
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT*2 , UNIT])#第一个阻塞格的中心位置
        #创建了一个矩形（障碍物），并将其填充为黑色
        #-15 和 +15 是为了使矩形的大小为 30x30 像素
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        #到达点
        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        
        #self.rect 创建了一个红色矩形，代表智能体的位置。
        #矩形的大小也是 30x30 像素。
        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        #         将之前创建的画布（canvas）添加到 Tkinter 窗口中并进行显示。
        # pack() 是 Tkinter 中的布局管理方法，它将画布控件添加到窗口并自动进行适当的布局
        self.canvas.pack()
    #重新初始化迷宫环境，通常在一个新的回合开始时调用
    def reset(self):
        self.update()
        time.sleep(0.5)
        #删除画布上的智能体矩形（self.rect）
        self.canvas.delete(self.rect)
        #设置智能体的新起始位置为 [20, 20]
        origin = np.array([20, 20])
        #重新创建了智能体的矩形
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        #返回智能体当前位置的坐标，坐标值是一个列表，格式为 [x1, y1, x2, y2]，代表矩形的左上角和右下角的坐标
        return self.canvas.coords(self.rect)
    
    #step() 方法用于在每次调用时执行一个动作并返回新的状态、奖励和是否完成的标志
    def step(self, action):
        #获取智能体当前的坐标，s 是一个列表，包含智能体矩形的左上角和右下角坐标 [x1, y1, x2, y2]。
        s = self.canvas.coords(self.rect)
        #初始化动作的变化量，base_action 表示智能体的坐标变化。它是一个二维数组 [dx, dy]，表示横向和纵向的变化
        base_action = np.array([0, 0])
        #判断是否是上移操作（action == 0 表示向上移动）
        if action == 0:   # up
            #确保当前智能体不是在最上方的一行（即 s[1] 代表智能体的顶部 y 坐标，确保它大于单位大小 UNIT 才能向上移动）。
            if s[1] > UNIT:
                #表示向上移动 UNIT 大小
                base_action[1] -= UNIT
        #判断是否是下移操作。
        elif action == 1:   # down
            #确保智能体不是在最下方的一行，s[1] 代表智能体顶部的 y 坐标，不能超过迷宫高度 (MAZE_H - 1) * UNIT。
            if s[1] < (MAZE_H - 1) * UNIT:
                #base_action[1] += UNIT 表示向下移动 UNIT 大小。
                base_action[1] += UNIT
        #判断是否是向右移动。
        elif action == 2:   # right
            #确保智能体不会越过迷宫的右边界。s[0] 是智能体矩形的左上角的 x 坐标，不能超过迷宫宽度 (MAZE_W - 1) * UNIT。
            if s[0] < (MAZE_W - 1) * UNIT:
                #表示向右移动 UNIT 大小。
                base_action[0] += UNIT
        #判断是否是向左移动。
        elif action == 3:   # left
        #确保智能体不会越过迷宫的左边界。s[0] 是智能体矩形的左上角的 x 坐标，不能小于 UNIT。
            if s[0] > UNIT:
                #向左移动 UNIT 大小。
                base_action[0] -= UNIT
        #移动智能体矩形到新的位置。base_action[0] 和 base_action[1] 分别表示 x 和 y 方向的移动量。
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        #获取智能体的新坐标，即移动后的状态 s_
        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        #智能体的坐标和目标区域的坐标相同
        if s_ == self.canvas.coords(self.oval):
            #reward = 1 给出奖励
            reward = 1
            #done = True 表示任务完成
            done = True
            #s_ = 'terminal' 标记当前状态为终止状态
            s_ = 'terminal'
        #如果智能体的坐标与地狱区域（障碍物）重合，表示智能体触碰到障碍物
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            #reward = -1 给出惩罚
            reward = -1
            #done = True 表示任务完成
            done = True
            # 标记当前状态为终止状态
            s_ = 'terminal'
        #智能体既没有到达目标区域也没有碰到障碍物
        else:
            # 表示没有奖励或惩罚
            reward = 0
            #任务未完成
            done = False
        #返回新的状态 s_，奖励 reward 和是否完成的标志 done
        return s_, reward, done
    
    def render(self):
        time.sleep(0.5)
            #self.update() 是 tkinter.Tk 类的方法。tkinter.Tk 继承自 Python 的 Tkinter 图形库
    # 它的 update() 方法用于刷新 GUI 界面。通常，self.update() 会在某些事件或界面变化后被调用，确保界面得到更新。
        self.update()

#这个循环执行 10 次，通常用来模拟 10 次回合的操作或 10 次环境更新。
def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

#创建并启动迷宫环境，调用 mainloop 进入 Tkinter 的事件循环。
if __name__ == '__main__':
    env = Maze()
    #after 方法用于安排一个延时操作，它的作用是将指定的函数（在这里是 update）在指定的时间（以毫秒为单位）之后执行。
    env.after(100, update)
    #一旦 mainloop() 被调用，它会进入一个无限循环，不断地响应用户的操作，更新界面，直到用户关闭窗口或者调用 quit() 方法为止。
    env.mainloop()
