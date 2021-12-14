from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch as t
import torch.nn as nn
import pendulum
import timeit
import RobotDART as rd



start = timeit.default_timer()

# configurations
env = pendulum.PendulumEnv()
env.__init__()
observe_dim = 3
action_num = 1
max_episodes = 1000
max_steps = 2500
solved_reward = -4000
solved_repeat = 5


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.softmax(self.fc3(a), dim=1)
        dist = Categorical(probs=probs)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


if __name__ == "__main__":
    actor = Actor(observe_dim, action_num)
    critic = Critic(observe_dim)

    ppo = PPO(actor, critic, t.optim.Adam, nn.MSELoss(reduction="sum"))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        tmp_observations = []
        k=0

        for k in range(10):
            state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
            expret = 0
            step = 0
            while not terminal and step <= max_steps:
                step += 1
                with t.no_grad():
                    old_state = state
                    # agent model inference
                    action = ppo.act({"state": old_state})[0]
                    state, reward, terminal, _ = env.step(action.numpy())
                    state = t.tensor(state, dtype=t.float32).view(1, observe_dim)

                    expret = expret + reward[0]

                    if k%10 == 0:
                        total_reward += reward[0]
                        tmp_observations.append(
                            {
                                "state": {"state": old_state},
                                "action": {"action": action},
                                "next_state": {"state": state},
                                "reward": reward[0],
                                "terminal": terminal or step == max_steps,
                            }
                        )
            with open("Data/Expected Return/PPO/PPO-ER1", 'a') as file:
                file.write(str(expret))
                file.write("\n")

        # update
        ppo.store_episode(tmp_observations)
        ppo.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        '''if smoothed_total_reward>-2000:
            graphics = rd.gui.Graphics()
            env.simu.set_graphics(graphics)
            graphics.look_at([0., 2.5, 0.5], [0., 0., 0.])'''

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")

                end = timeit.default_timer()
                time = str(end-start)
                with open("Data/PPO-Time", 'a') as file:
                    file.write(time)
                    file.write("\n")

                exit(0)
        else:
            reward_fulfilled = 0