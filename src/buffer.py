from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class ReplayBuffer:
    def __init__(self, data_spec, batch_size=1, smaple_batch_size=64):
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.smaple_batch_size = smaple_batch_size
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size)

    def __len__(self):
        return self.buffer.num_frames()

    def collect(self, env, policy):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.from_transition(
            time_step, action_step, next_time_step)
        self.buffer.add_batch(traj)

    def get_pipeline(self, num_parallel_calls=3, num_steps=2, num_prefetch=3):
        dataset = self.buffer.as_dataset(
            num_parallel_calls=num_parallel_calls,
            sample_batch_size=self.smaple_batch_size,
            num_steps=num_steps).prefetch(num_prefetch)
        return iter(dataset)
