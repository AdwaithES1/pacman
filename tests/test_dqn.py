import numpy as np
import pytest
import torch

from rl.agents import DQNAgent, DQNConfig, QNetwork, ReplayBuffer
from rl.training import train as training

torch.set_num_threads(1)


OBSERVATION_SHAPE = (6, 31, 28)


def _config(**overrides):
    values = {
        "batch_size": 2,
        "replay_capacity": 8,
        "warmup_steps": 0,
        "target_update_interval": 1,
        "hidden_size": 16,
        "conv_channels": (4,),
        "device": "cpu",
    }
    values.update(overrides)
    return DQNConfig(**values)


def test_q_network_accepts_single_and_batched_observations():
    network = QNetwork(OBSERVATION_SHAPE, action_count=4, conv_channels=(4,), hidden_size=16)
    observation = torch.zeros(OBSERVATION_SHAPE)
    assert network(observation).shape == (1, 4)
    assert network(observation.unsqueeze(0).repeat(3, 1, 1, 1)).shape == (3, 4)


def test_q_network_validates_shape_and_normalization():
    network = QNetwork(OBSERVATION_SHAPE, action_count=4, conv_channels=(4,), hidden_size=16)
    with pytest.raises(ValueError, match="Expected observations"):
        network(torch.zeros(1, 2, 10, 10))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        network(torch.full((1, *OBSERVATION_SHAPE), 2.0))


def test_replay_buffer_samples_typed_transitions_with_both_end_flags():
    buffer = ReplayBuffer(3, OBSERVATION_SHAPE, seed=11)
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    next_state = np.ones(OBSERVATION_SHAPE, dtype=np.float32)
    buffer.add(state, 2, 1.5, next_state, terminated=False, truncated=True)
    batch = buffer.sample(1)
    assert batch.states.shape == (1, *OBSERVATION_SHAPE)
    assert batch.states.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.terminated.tolist() == [False]
    assert batch.truncated.tolist() == [True]
    assert batch.next_states[0, 0, 0, 0] == 1.0


def test_replay_buffer_sampling_is_reproducible_and_wraps():
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    buffer_a = ReplayBuffer(2, OBSERVATION_SHAPE, seed=3)
    buffer_b = ReplayBuffer(2, OBSERVATION_SHAPE, seed=3)
    for action in range(3):
        buffer_a.add(state, action, float(action), state, False, False)
        buffer_b.add(state, action, float(action), state, False, False)
    assert len(buffer_a) == 2
    assert buffer_a.sample(2).actions.tolist() == buffer_b.sample(2).actions.tolist()


def test_action_selection_is_valid_and_epsilon_greedy():
    agent = DQNAgent(OBSERVATION_SHAPE, 4, _config(seed=19))
    observation = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        agent.online_network.head[-1].bias.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert agent.select_action(observation, explore=False) == 3
    assert agent.select_action(observation, epsilon=0.0) == 3
    actions = {agent.select_action(observation, epsilon=1.0) for _ in range(30)}
    assert len(actions) > 1
    assert all(0 <= action < 4 for action in actions)


def test_td_targets_mask_termination_but_bootstrap_truncation():
    rewards = torch.tensor([2.0, 2.0])
    next_q_values = torch.tensor([10.0, 10.0])
    terminated = torch.tensor([True, False])
    targets = DQNAgent.compute_td_targets(rewards, next_q_values, terminated, gamma=0.5)
    assert targets.tolist() == [2.0, 7.0]


def test_dqn_performs_gradient_update_after_warmup():
    agent = DQNAgent(OBSERVATION_SHAPE, 4, _config())
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    for index in range(2):
        agent.replay_buffer.add(state, index, 1.0, state, index == 0, index == 1)
    agent.global_step = 1
    loss = agent.learn()
    assert loss is not None
    assert np.isfinite(loss)
    assert agent.gradient_updates == 1
    for online, target in zip(agent.online_network.parameters(), agent.target_network.parameters()):
        assert torch.equal(online, target)


def test_dqn_waits_for_both_warmup_and_full_batch():
    agent = DQNAgent(
        OBSERVATION_SHAPE,
        4,
        _config(batch_size=4, replay_capacity=8, warmup_steps=3, target_update_interval=10),
    )
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    for action in range(4):
        agent.replay_buffer.add(state, action, 1.0, state, False, False)

    agent.global_step = 2
    assert agent.learn() is None
    assert agent.gradient_updates == 0

    agent.global_step = 3
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)
    assert agent.gradient_updates == 1

    second_agent = DQNAgent(
        OBSERVATION_SHAPE,
        4,
        _config(batch_size=4, replay_capacity=8, warmup_steps=0),
    )
    for action in range(3):
        second_agent.replay_buffer.add(state, action, 1.0, state, False, False)
    second_agent.global_step = 20
    assert second_agent.learn() is None
    assert second_agent.gradient_updates == 0


def test_target_sync_interval_counts_gradient_updates_not_environment_steps():
    agent = DQNAgent(
        OBSERVATION_SHAPE,
        4,
        _config(batch_size=1, replay_capacity=4, warmup_steps=0, target_update_interval=2),
    )
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    agent.replay_buffer.add(state, 0, 1.0, state, False, False)
    agent.global_step = 1_000
    agent.learn()
    assert agent.gradient_updates == 1
    assert any(
        not torch.equal(online, target)
        for online, target in zip(agent.online_network.parameters(), agent.target_network.parameters())
    )
    agent.learn()
    assert agent.gradient_updates == 2
    assert all(
        torch.equal(online, target)
        for online, target in zip(agent.online_network.parameters(), agent.target_network.parameters())
    )


def test_epsilon_decays_with_global_environment_steps():
    agent = DQNAgent(OBSERVATION_SHAPE, 4, _config(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=10))
    assert agent.epsilon() == 1.0
    agent.global_step = 5
    assert agent.epsilon() == 0.5
    agent.global_step = 20
    assert agent.epsilon() == 0.0


def test_cli_values_override_dqn_configuration_defaults():
    args = training._parser().parse_args([
        "--batch-size", "16",
        "--replay-capacity", "800",
        "--warmup-steps", "128",
        "--epsilon-decay-steps", "4000",
        "--target-update-interval", "75",
        "--learning-rate", "0.0003",
        "--device", "cpu",
    ])
    config = training._config_from_args(args)

    assert config.batch_size == 16
    assert config.replay_capacity == 800
    assert config.warmup_steps == 128
    assert config.epsilon_decay_steps == 4000
    assert config.target_update_interval == 75
    assert config.learning_rate == 0.0003
    assert config.device == "cpu"


def test_checkpoint_round_trip_restores_agent_and_replay(tmp_path):
    agent = DQNAgent(OBSERVATION_SHAPE, 4, _config(seed=23))
    state = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)
    agent.replay_buffer.add(state, 1, 2.0, state, False, True)
    agent.global_step = 5
    agent.completed_episodes = 2
    checkpoint = tmp_path / "checkpoint.pt"
    agent.save_checkpoint(checkpoint)

    restored = DQNAgent.load_checkpoint(checkpoint, OBSERVATION_SHAPE, 4, device="cpu")
    assert restored.global_step == 5
    assert restored.completed_episodes == 2
    assert len(restored.replay_buffer) == 1
    assert restored.replay_buffer.truncated[0]
    for original, loaded in zip(agent.online_network.parameters(), restored.online_network.parameters()):
        assert torch.equal(original, loaded)


def test_checkpoint_rejects_incompatible_environment(tmp_path):
    agent = DQNAgent(OBSERVATION_SHAPE, 4, _config())
    checkpoint = tmp_path / "checkpoint.pt"
    agent.save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="does not match"):
        DQNAgent.load_checkpoint(checkpoint, OBSERVATION_SHAPE, 3, device="cpu")
