from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class ReplayBuffer:
    def __init__(self, data_spec, batch_size=1, epochs=32):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.epochs = epochs
        self.replay_buffer_capacity = 1000000
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=self.replay_buffer_capacity)

    def __len__(self):
        return self.buffer.num_frames()

    def clear(self):
        return self.buffer.clear()

    def gather_all(self):
        return self.buffer.gather_all()

    def collect(self, env, policy):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(
            time_step, action_step, next_time_step)
        self.buffer.add_batch(traj)
        return traj

    def collect_episode(self, env, policy):
        counter = 0
        while counter < self.epochs:
            print(len(self))
            traj = self.collect(env, policy)
            if traj.is_boundary():
                counter += 1
