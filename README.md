# Pac-Man RL Integration

This project combines the existing browser Pac-Man game with a Python Gymnasium-compatible environment for future reinforcement learning work. The browser remains the source of game mechanics while Python consumes live state and uses the local HTTP API for reset/step actions.

## Architecture

```text
Browser Pac-Man Game (JavaScript)
       |
       | HTTP API / SSE
       v
Python Game Client
       |
       v
Gymnasium Environment
       |
       v
Observation + Reward Processing
       |
       v
CNN DQN Agent + Replay Buffer
```

## Current implementation status

Implemented this phase:
- JavaScript server exposes maze metadata and live state snapshots
- Local agent-control endpoints for reset and step
- Python HTTP client for the game server
- Gymnasium environment with a 6-channel 28x31 observation tensor
- Reward calculation helpers
- Automated tests for rewards and environment integration
- CNN-based DQN with target network, replay, CSV metrics, and checkpoints

Not yet implemented:
- Double DQN or prioritized replay
- A claim of successful policy learning (short smoke runs only validate the pipeline)
- Advanced ghost-policy optimization beyond the available state API

## Directory structure

```text
pacman-tracker/
├── public/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── rl/
│   ├── __init__.py
│   ├── agents/
│   │   ├── dqn.py
│   │   └── replay_buffer.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── game_client.py
│   │   ├── observations.py
│   │   ├── pacman_env.py
│   │   └── rewards.py
│   ├── training/
│   │   ├── metrics.py
│   │   ├── smoke_test.py
│   │   └── train.py
│   └── utils/
├── tests/
│   ├── test_environment.py
│   ├── test_game_integration.py
│   └── test_rewards.py
├── README.md
├── package.json
├── requirements.txt
├── run_training.bat
├── server.js
├── start.bat
└── public/
```

## Prerequisites

- Node.js 18+
- Python 3.10+
- PyTorch 2.2+
- Matplotlib 3.8+
- A local, trusted network environment (the agent API binds to localhost by default)

## Installation

### JavaScript dependencies

```powershell
npm install
```

### Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Choose a PyTorch build suitable for your machine when needed; training defaults to CUDA when available and otherwise uses CPU. Use `--device cpu` or `--device cuda` to select explicitly.

## Start the game server

```powershell
node server.js
```

or use the Windows launcher:

```powershell
.\start.bat
```

Then open http://localhost:3000 in your browser.

## Local API

The game exposes the following endpoints:

- GET /api/health
- GET /api/maze
- GET /api/state
- POST /api/state
- GET /api/stream
- GET /api/history?limit=120
- GET /api/events
- POST /api/agent/reset
- POST /api/agent/step
- GET /api/agent/state

### Agent API contract

Reset:

```http
POST /api/agent/reset
Content-Type: application/json

{"max_steps": 200}
```

Response:

```json
{
  "ok": true,
  "state": { ... },
  "reward": 0,
  "terminated": false,
  "truncated": false,
  "info": { "maxSteps": 200, "reset": true }
}
```

Step:

```http
POST /api/agent/step
Content-Type: application/json

{"action": 0}
```

Action mapping:

- 0 = UP
- 1 = DOWN
- 2 = LEFT
- 3 = RIGHT

The server validates action values and rejects invalid requests with an HTTP 400 response.

## Observation format

The Python environment uses a deterministic 6-channel tensor shaped `(6, 31, 28)`. It reads the static wall map from `/api/maze` and the live pellet/entity positions from the agent state:

1. Walls
2. Remaining regular pellets
3. Remaining power pellets
4. Pac-Man position
5. Dangerous ghost positions
6. Frightened ghost positions

Coordinate convention:
- Origin: top-left tile
- `x` increases to the right
- `y` increases downward
- Based on the actual maze reported by the server rather than a hard-coded assumption

## Reward function

Reward rules are configurable via `RewardConfig` in `rl/environment/rewards.py`.

Implemented defaults:

- Regular pellet: +1.0
- Power pellet: +3.0
- Ghost capture while frightened: +5.0
- Life lost: -10.0
- Maze clear: +50.0
- All lives lost: -20.0

Rewards are cumulative across events in the same simulation step. No arbitrary per-step penalty is added in this phase.

There is no time or movement penalty. Ghost-capture reward is recognized from the agent API's `ghost-captured` event (the current game model awards +5); losing a life is -10 and losing the last life adds -20 more. This sparse event reward is usable but does not reward progress through empty corridors, so the short smoke results do not establish that it is informative enough for reliable learning. Keep the reward unchanged until longer controlled experiments show a consistent issue.

The JavaScript game keeps an actual game score (`state.score`), while the Gymnasium environment exposes a per-step RL reward (`reward` / `info["step_reward"]`) and a running episode return (`info["episode_return"]`). These values are intentionally tracked separately so training logs can report both the score and the RL reward without conflating them.

## Terminal conditions

The game agent treats one maze as one episode.

- Win: all pellets are cleared
- Loss: Pac-Man has no lives remaining
- Truncation: max step limit reached before termination

The RL episode limit is configured via the environment `max_steps` value and is deliberately separate from the browser game's own level flow. The game continues its normal automatic level progression in the browser; the RL environment only truncates its own training episode when the configured step limit is reached.

The environment does not automatically reset after a terminal state; the training and evaluation loops call `reset()` explicitly between episodes.

DQN treats a true win or game-over as terminal and does not bootstrap its target from the next observation. A max-step truncation is not terminal: the valid final observation returned by the environment is stored before reset and the learner bootstraps from it. This is the time-limit policy used consistently by the replay buffer, update, and training loop.

## Running tests

```powershell
python -m pytest -q
```

or use the Windows launcher:

```powershell
.\run_training.bat
```

This runs `pytest` for the Python environment, DQN, replay buffer, checkpoint, and metrics logic. The live integration tests also start the local JavaScript server when Node.js is available.

## Smoke test

Start the game server, then run the end-to-end RL smoke-test script:

```powershell
node server.js
```

Then, in a second PowerShell window:

```powershell
python -m rl.training.smoke_test
```

If you want a quick health check first:

```powershell
curl http://localhost:3000/api/health
```

or:

```powershell
python -c "import requests; print(requests.get('http://127.0.0.1:3000/api/health').json())"
```

### Windows smoke-test instructions

1. Open PowerShell in the project root.
2. Install Node.js dependencies if needed:
   ```powershell
   npm install
   ```
3. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install Python dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
5. Start the JavaScript game server:
   ```powershell
   node server.js
   ```
6. Run the integration smoke test from a second terminal:
   ```powershell
   python -m rl.training.smoke_test
   ```
7. If the server is not running, the script exits with a clear error instead of silently using mock data.

## DQN training

Start the JavaScript server in one PowerShell window from the project root:

```powershell
node server.js
```

In a second PowerShell window, run a short real-game pipeline check:

```powershell
python -m rl.training.train --total-steps 16 --max-steps 8 --batch-size 4 --replay-capacity 64 --warmup-steps 4 --target-update-interval 2 --output-root runs --device cpu
```

This executes actual reset/step calls against the server, performs gradient updates after replay warm-up, records completed episodes, and saves a checkpoint. It is only an integration smoke run; it is not evidence that the policy has learned to play well.

Useful training options include `--total-steps`, `--max-episodes`, `--max-steps` (RL time limit per maze attempt), `--seed`, `--device`, `--learning-rate`, `--gamma`, `--batch-size`, `--replay-capacity`, `--warmup-steps`, `--epsilon-start`, `--epsilon-end`, `--epsilon-decay-steps`, and `--target-update-interval`. Defaults are 10,000 environment steps, 200 steps per episode, seed 7, Adam learning rate `1e-4`, discount `0.99`, batch size 32, replay capacity 50,000, 1,000 warm-up steps, epsilon linearly decaying from 1.0 to 0.05 over 20,000 environment steps, and a hard target update every 500 gradient updates. A gradient update requires both the global step warm-up and at least one full batch in replay; epsilon is based on global environment steps, while target synchronization is based on gradient-update count.

By default, training saves stage checkpoints every 2,500 requested environment steps and greedily evaluates them every 5,000 steps using five episodes. These triggers run at episode boundaries, so the recorded step may be up to one episode limit past the requested interval. Stage evaluations share the training episode limit and a fixed evaluation seed, and do not update weights or optimizer state. Configure `--checkpoint-every-steps`, `--evaluation-every-steps`, `--checkpoint-evaluation-episodes`, and `--evaluation-seed`; set either interval to `0` to disable it. The standalone `--eval-episodes` default is 20.

Each invocation creates a new uniquely named directory under `runs` (or the selected `--output-root`) with:

- `config.json`: DQN configuration, environment limits, device, reward setup, and software versions.
- `run_metadata.json`: experiment ID, start/finish times, run status, software versions, environment endpoint, and counters.
- `episodes.csv`: completed-episode return (sum of environment step rewards), actual game score, pellets, lives lost, duration, end reason, epsilon, latest loss, and update count.
- `training.csv`: periodic rolling return/score/pellet/completion summaries, loss, epsilon, learning rate, update count, throughput, and elapsed time.
- `checkpoint.pt`: online/target weights, optimizer, counters, replay buffer, and random-number-generator states.
- `checkpoints\step-*.pt`: persistent stage checkpoints used for progress evaluations.
- `evaluations\<evaluation-run>\`: stage-wise greedy evaluation metrics kept separate from training metrics.

To resume training, use the saved checkpoint and choose a new total environment-step target:

```powershell
python -m rl.training.train --resume runs\<run-directory>\checkpoint.pt --total-steps 20000 --max-steps 200 --output-root runs
```

The checkpoint's DQN hyperparameters are restored when resuming. `--total-steps` is the target global environment step, not an additional step count.

## Evaluation, baseline, plots, and reports

Run evaluation and baseline commands from the project root while the JavaScript server is running. Evaluation uses greedy actions with no epsilon exploration, and does not update model or optimizer weights. The random policy samples only from the environment action space. Both use the same default reward behavior and configured RL time limit.

Evaluate a saved checkpoint in 20 episodes, each limited to 200 environment steps (override `--episodes` to configure the count):

```powershell
python -m rl.training.experiment_workflow evaluate --checkpoint runs\<run-directory>\checkpoint.pt --episodes 20 --max-steps 200 --seed 7 --output-root evaluations
```

Run the random-action baseline with matching evaluation conditions:

```powershell
python -m rl.training.experiment_workflow random --episodes 20 --max-steps 200 --seed 7 --output-root evaluations
```

Each command writes a new output directory containing `config.json`, `run_metadata.json`, `episodes.csv`, and aggregate `summary.json`. Mean and standard deviation are reported for recorded numeric metrics. Unknown metrics remain unavailable; a completion rate is only computed from episodes with an explicit win/loss result.

Compare the saved greedy DQN evaluation and random-policy baseline:

```powershell
python -m rl.training.experiment_workflow compare --dqn evaluations\<dqn-evaluation-directory> --baseline evaluations\<random-baseline-directory> --output-root comparisons
```

The comparison records condition mismatches and seed differences and does not infer a performance advantage.

Generate PNG and PDF plots from the existing training and optional evaluation results (rolling smoothing defaults to 10 episodes):

```powershell
python -m rl.training.experiment_workflow plot --experiment runs\<training-run-directory> --evaluation evaluations\<dqn-evaluation-directory> --baseline evaluations\<random-baseline-directory> --smoothing-window 10
```

Generate a preliminary Markdown report and its plots using only previously saved artifacts:

```powershell
python -m rl.training.experiment_workflow report --experiment runs\<training-run-directory> --evaluation evaluations\<dqn-evaluation-directory> --baseline evaluations\<random-baseline-directory> --output-root reports --smoothing-window 10
```

Generated plot and report directories are unique; existing experiment output, metrics, and checkpoints are not overwritten. Missing metrics are shown as unavailable or as empty plot series, not imputed. Reports with short runs or no evaluation are marked preliminary. Plots include separate RL return and actual game score, episode return smoothing, pellets, episode length, training loss, epsilon, gradient updates, checkpoint-progress evaluations, DQN and random evaluation metrics, and PDF versions.

Training and evaluation checkpoints should only be loaded from trusted sources because PyTorch checkpoint files use Python serialization.

## Current limitations

- The environment currently relies on the local game server and a small training-state model rather than a full ROM-accurate emulation.
- DQN is a baseline CNN agent; the reporting workflow does not make a learning claim from smoke-test-scale runs.
- The browser game remains the authoritative game engine; the Python side does not reimplement movement or collision logic.
- The control API is intentionally local-only to avoid exposing a training interface unnecessarily.
