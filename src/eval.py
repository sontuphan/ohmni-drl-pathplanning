import matplotlib.pyplot as plt


class ExpectedReturn:
    def __init__(self):
        self.returns = None

    def compute_avg_return(self, tfenv, agent, num_episodes):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = tfenv.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = agent.action(time_step)
                time_step = tfenv.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def eval(self, tfenv, agent, num_episodes=5):
        avg_return = self.compute_avg_return(tfenv, agent, num_episodes)
        if self.returns is None:
            self.returns = [avg_return]
        else:
            self.returns.append(avg_return)
        return avg_return

    def save(self):
        plt.plot(self.returns)
        plt.ylabel('Average Return')
        plt.savefig('models/eval.jpg')
