import gym
from agent import Agent
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
TEST_EPISODES = 20


def main():
    test_env = gym.make(ENV_NAME)
    agent = Agent(ENV_NAME)
    writer = SummaryWriter(comment="-value-iteration")

    iter_no = 0
    best_reward = 0.0

    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated {:.3f} -> {:.3f}".format(best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in {} iterations.".format(iter_no))
            break
    writer.close()


if __name__ == '__main__':
    main()
