'use strict';

(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $('gameCanvas');
  const ctx = canvas.getContext('2d');
  const controls = ['startBtn', 'pauseBtn', 'resumeBtn', 'stopBtn'];
  let playbackState = null;

  function setMessage(message, error = false) {
    $('message').textContent = message;
    $('message').classList.toggle('error', error);
  }

  async function request(path, payload) {
    const response = await fetch(path, {
      method: payload === undefined ? 'GET' : 'POST',
      headers: payload === undefined ? {} : { 'Content-Type': 'application/json' },
      body: payload === undefined ? undefined : JSON.stringify(payload),
      cache: 'no-store',
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    return result;
  }

  function drawBoard(state) {
    if (!state || !Array.isArray(state.maze?.rows) || !Array.isArray(state.pelletRows)) {
      $('emptyBoard').hidden = false;
      return;
    }
    $('emptyBoard').hidden = true;
    const width = canvas.width;
    const height = canvas.height;
    const cell = width / state.maze.width;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#040817';
    ctx.fillRect(0, 0, width, height);

    for (let y = 0; y < state.maze.rows.length; y += 1) {
      const mazeRow = state.maze.rows[y];
      const pelletRow = state.pelletRows[y] || '';
      for (let x = 0; x < mazeRow.length; x += 1) {
        const px = x * cell;
        const py = y * cell;
        if (mazeRow[x] === '#') {
          ctx.fillStyle = '#07143f';
          ctx.strokeStyle = '#3472ff';
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.roundRect(px + 1.2, py + 1.2, cell - 2.4, cell - 2.4, 4);
          ctx.fill();
          ctx.stroke();
        } else if (mazeRow[x] === '-') {
          ctx.strokeStyle = '#ffbfff';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(px + 2, py + cell / 2);
          ctx.lineTo(px + cell - 2, py + cell / 2);
          ctx.stroke();
        } else if (pelletRow[x] === '.' || pelletRow[x] === 'o') {
          ctx.fillStyle = pelletRow[x] === 'o' ? '#ffe58b' : '#e7ecfa';
          ctx.beginPath();
          ctx.arc(px + cell / 2, py + cell / 2, pelletRow[x] === 'o' ? cell * 0.25 : cell * 0.085, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    const entities = Array.isArray(state.entities) ? state.entities : [];
    for (const entity of entities) {
      const cx = Number(entity.x) * cell + cell / 2;
      const cy = Number(entity.y) * cell + cell / 2;
      const radius = cell * 0.41;
      if (entity.role === 'player' || entity.id === 'pacman') {
        const angles = { up: -Math.PI / 2, down: Math.PI / 2, left: Math.PI, right: 0 };
        const angle = angles[entity.direction] ?? Math.PI;
        ctx.fillStyle = '#ffdf3c';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, radius, angle + 0.34, angle + Math.PI * 2 - 0.34);
        ctx.closePath();
        ctx.fill();
      } else {
        ctx.fillStyle = entity.state === 'frightened' ? '#3166ff' : (entity.color || '#ff4e61');
        ctx.beginPath();
        ctx.arc(cx, cy, radius, Math.PI, 0);
        ctx.lineTo(cx + radius, cy + radius * 0.8);
        ctx.lineTo(cx + radius * 0.5, cy + radius * 0.45);
        ctx.lineTo(cx, cy + radius * 0.8);
        ctx.lineTo(cx - radius * 0.5, cy + radius * 0.45);
        ctx.lineTo(cx - radius, cy + radius * 0.8);
        ctx.closePath();
        ctx.fill();
        if (entity.state !== 'frightened' && entity.state !== 'eyes') {
          ctx.fillStyle = '#f7fbff';
          ctx.beginPath();
          ctx.ellipse(cx - radius * 0.34, cy - radius * 0.12, radius * 0.19, radius * 0.24, 0, 0, Math.PI * 2);
          ctx.ellipse(cx + radius * 0.34, cy - radius * 0.12, radius * 0.19, radius * 0.24, 0, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  function updateControls(state) {
    const status = state.status;
    $('startBtn').disabled = ['loading', 'playing', 'paused', 'stopping'].includes(status);
    $('pauseBtn').disabled = status !== 'playing';
    $('resumeBtn').disabled = status !== 'paused';
    $('stopBtn').disabled = !['loading', 'playing', 'paused', 'stopping', 'won', 'lost', 'truncated', 'error'].includes(status);
    for (const id of controls) $(id).setAttribute('aria-busy', status === 'loading' || status === 'stopping' ? 'true' : 'false');
  }

  function render(state) {
    playbackState = state;
    const status = state.status || 'stopped';
    const badge = $('runStatus');
    badge.textContent = status.toUpperCase();
    badge.className = `status-badge ${status}`;
    $('scoreValue').textContent = String(state.game_score ?? 0);
    $('pelletValue').textContent = `${state.pellets_collected ?? 0} / ${state.pellets_total ?? 0}`;
    $('stepValue').textContent = String(state.steps ?? 0);
    $('livesValue').textContent = state.lives == null ? '—' : String(state.lives);
    $('actionValue').textContent = state.current_action?.name || '—';
    $('rewardValue').textContent = Number(state.step_reward ?? 0).toFixed(2);
    $('returnValue').textContent = Number(state.episode_return ?? 0).toFixed(2);
    $('episodeValue').textContent = String(state.episode_status || 'not_started').replaceAll('_', ' ').toUpperCase();
    $('checkpointValue').textContent = state.checkpoint || '—';
    $('observationValue').textContent = `(${(state.observation_shape || [6, 31, 28]).join(', ')})`;
    $('speedInput').value = String(state.speed_ms ?? 500);
    $('speedValue').textContent = `${state.speed_ms ?? 500} ms / action`;
    drawBoard(state.game_state);
    updateControls(state);

    if (state.error) setMessage(state.error, true);
    else if (status === 'loading') setMessage('Loading checkpoint and claiming the isolated game session…');
    else if (status === 'playing') setMessage('The greedy DQN policy is controlling the Gymnasium environment.');
    else if (status === 'paused') setMessage('Paused. The current episode state is preserved.');
    else if (status === 'stopping') setMessage('Stopping and releasing the agent session…');
    else if (['won', 'lost', 'truncated'].includes(status)) setMessage(`Episode ${status}. Session resources have been released.`);
    else if (status === 'error') setMessage('Playback failed. See the error above, then correct it and retry.', true);
    else setMessage('Ready to load the trained policy.');
  }

  async function refresh() {
    try {
      const state = await request('/api/playback/state');
      $('connectionText').textContent = 'VIEWER ONLINE';
      $('connectionDot').classList.add('online');
      $('connectionDot').parentElement.classList.add('online');
      render(state);
    } catch (error) {
      $('connectionText').textContent = 'BACKEND OFFLINE';
      $('connectionDot').parentElement.classList.remove('online');
      setMessage(error.message, true);
    }
  }

  async function command(path, payload, successMessage) {
    setMessage(successMessage);
    try {
      const state = await request(path, payload);
      render(state);
    } catch (error) {
      setMessage(error.message, true);
      await refresh();
    }
  }

  $('startBtn').addEventListener('click', () => command('/api/playback/start', {}, 'Loading checkpoint and starting playback…'));
  $('pauseBtn').addEventListener('click', () => command('/api/playback/pause', {}, 'Pausing playback…'));
  $('resumeBtn').addEventListener('click', () => command('/api/playback/resume', {}, 'Resuming playback…'));
  $('stopBtn').addEventListener('click', () => command('/api/playback/stop', {}, 'Stopping playback and releasing resources…'));

  $('speedInput').addEventListener('input', () => {
    $('speedValue').textContent = `${$('speedInput').value} ms / action`;
  });
  $('speedInput').addEventListener('change', () => {
    command('/api/playback/speed', { interval_ms: Number($('speedInput').value) }, 'Updating playback speed…');
  });

  render({
    status: 'stopped',
    episode_status: 'not_started',
    checkpoint: 'runs/learning-seed73-20261014/pacman-dqn-20261014T133045092713Z-seed73-2ae528fa/checkpoint.pt',
    observation_shape: [6, 31, 28],
    speed_ms: 500,
    game_state: null,
  });
  refresh();
  window.setInterval(refresh, 250);
})();
