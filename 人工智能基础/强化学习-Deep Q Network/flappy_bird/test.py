import argparse
import game.wrapped_flappy_bird as game
import torch.cuda
from utils import resize_and_bgr2gray, image_to_tensor


def get_args():
    parser = argparse.ArgumentParser("""使用DQN玩flappy bird""")
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def test(opt):
    #初始化
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # 读取模型
    if torch.cuda.is_available():
        model = torch.load("{}/flappy_bird".format(opt.saved_path))
    else:
        model = torch.load("{}/flappy_bird".format(opt.saved_path), map_location=lambda storage, loc:storage)

    # 使设置评估模式
    model.eval()

    # 初始化游戏环境
    game_state = game.GameState()
    # 定义一开始的动作，其实就是do nothing
    action = torch.zeros(2, dtype=torch.float32)
    action[0] = 1
    # 让游戏开始去运行 得到初始化的图片数据，奖励，是否结束
    image_data, reward, terminal = game_state.frame_step(action)

    # 对图片进行预处理
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    if torch.cuda.is_available():
        image_data = image_data.cuda()
        model.cuda()

    # 得到初始状态
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # 得到当前状态对应的Q值
        prediction = model(state)[0]
        action_index = torch.argmax(prediction).item()
        # 构造需要的action
        action = torch.zeros(2, dtype=torch.float32)
        action[action_index] = 1

        # 执行行为，得到下一个状态的图片数据，奖励，是否结束
        image_data, reward, terminal = game_state.frame_step(action)
        # 对图片进行预处理
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)

        if torch.cuda.is_available():
            image_data = image_data.cuda()
        #组合得到下一个状态
        next_state = torch.cat((state.squeeze(0)[1:, :, :], image_data)).unsqueeze(0)
        state = next_state


if __name__ == '__main__':
    args = get_args()
    test(args)
