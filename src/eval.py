import imageio
import matplotlib.pyplot as plt


class ExpectedReturn:
    def __init__(self):
        self.returns = None

    def compute_avg_return(self, env, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def eval(self, env, policy, num_episodes):
        avg_return = self.compute_avg_return(env, policy, num_episodes)
        if self.returns is None:
            self.returns = [avg_return]
        else:
            self.returns.append(avg_return)
        return avg_return

    def display(self, num_iterations, eval_interval):
        iterations = range(0, num_iterations + 1, eval_interval)
        plt.plot(iterations, self.returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.show()

    def create_policy_eval_video(self, env, display, policy, filename, num_episodes=5):
        filename = filename + ".mp4"
        with imageio.get_writer(filename, fps=24) as video:
            for _ in range(num_episodes):
                time_step = env.reset()
                video.append_data(display.render())
                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = env.step(action_step.action)
                    video.append_data(display.render())
