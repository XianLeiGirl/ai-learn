import numpy as np
import matplotlib.pyplot as plt

# 代理通过探索和利用策略学习Q值，最终能够找到从起点(0,0)到终点(3,3)的最优路径

# 定义网格世界环境
class GridWorld:
    def __init__(self, size):
        self.size = size
        self.state = (0, 0)  # 初始状态
        self.end_state = (size-1, size-1)  # 终点状态
        self.actions = ['up', 'down', 'left', 'right']
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.size - 1:
            y += 1
        self.state = (x, y)
        if self.state == self.end_state:
            return self.state, 1, True
        else:
            return self.state, -0.1, False  # 给一个小惩罚以鼓励更快到达终点

# Q学习算法实现
def q_learning(env, episodes, alpha, gamma, epsilon):
    q_table = np.zeros((env.size, env.size, len(env.actions)))
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(env.actions)  # 探索
            else:
                action = env.actions[np.argmax(q_table[state[0], state[1], :])]  # 利用
            next_state, reward, done = env.step(action)
            q_table[state[0], state[1], env.actions.index(action)] = \
                q_table[state[0], state[1], env.actions.index(action)] + \
                alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1], :]) - q_table[state[0], state[1], env.actions.index(action)])
            state = next_state
    return q_table

# 可视化Q值
def plot_q_table(q_table):
    fig, ax = plt.subplots()
    for x in range(q_table.shape[0]):
        for y in range(q_table.shape[1]):
            for action in range(q_table.shape[2]):
                if q_table[x, y, action] != 0:
                    ax.text(y, x, f'{env.actions[action]}:\n{q_table[x, y, action]:.2f}', ha='center', va='center', color='black')
    plt.imshow(np.max(q_table, axis=2), cmap='hot', interpolation='nearest')
    plt.colorbar(label='Q-value')
    plt.show()

# 主程序
env = GridWorld(4)
q_table = q_learning(env, 1000, 0.1, 0.9, 0.1)
plot_q_table(q_table)

