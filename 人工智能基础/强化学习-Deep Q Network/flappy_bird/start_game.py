import game.wrapped_flappy_bird as game
import numpy as np
import pygame

# 我们有两种可选的行为，一种就是不动，飞翔的小鸟就会自己下落
# 另一种就是我们点击比如鼠标的左键，小鸟就向上飞一下
ACTIONS = 2


def play_game():
    # 启动游戏，通过类创建了一个实例对象
    game_state = game.GameState()

    # while True 死循环
    while True:
        a_t = np.zeros([ACTIONS])  # array([0., 0.])
        # array([1., 0.]) 代表就是下降，相当于鼠标没有点击，也等同于 do nothing
        # array([0., 1.]) 代表着就是小鸟上升，背后相当于鼠标点击左键
        # 默认就往下下降，除非我们点击鼠标了
        a_t[0] = 1  # array([1., 0.])

        # 我们去检测我们是否真的点击了鼠标
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                a_t = np.zeros([ACTIONS])
                a_t[1] = 1  # array([0., 1.])
            else:
                pass

        # 最重要的就是调用下面的方法，它需要传入输入的actions
        _, _, terminal = game_state.frame_step(a_t)

        if terminal:
            break #不break会重新来


def main():
    play_game() 


if __name__ == '__main__':
    main()
