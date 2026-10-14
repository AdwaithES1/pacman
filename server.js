'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = __dirname;
const PUBLIC_DIR = path.join(ROOT, 'public');
const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '127.0.0.1';
const MAX_BODY = 2 * 1024 * 1024;
const HISTORY_LIMIT = 240;
const EVENT_LIMIT = 80;

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const MAZE_ROWS = [
  '############################',
  '#............##............#',
  '#.####.#####.##.#####.####.#',
  '#o####.#####.##.#####.####o#',
  '#.####.#####.##.#####.####.#',
  '#..........................#',
  '#.####.##.########.##.####.#',
  '#.####.##.########.##.####.#',
  '#......##....##....##......#',
  '######.##### ## #####.######',
  '     #.##### ## #####.#     ',
  '     #.##          ##.#     ',
  '     #.## ###--### ##.#     ',
  '######.## #      # ##.######',
  '          #      #          ',
  '######.## #      # ##.######',
  '     #.## ######## ##.#     ',
  '     #.##          ##.#     ',
  '     #.## ######## ##.#     ',
  '######.## ######## ##.######',
  '#............##............#',
  '#.####.#####.##.#####.####.#',
  '#.####.#####.##.#####.####.#',
  '#o..##................##..o#',
  '###.##.##.########.##.##.###',
  '###.##.##.########.##.##.###',
  '#......##....##....##......#',
  '#.##########.##.##########.#',
  '#.##########.##.##########.#',
  '#..........................#',
  '############################',
];

function isGhostHouseTile(x, y) {
  return x >= 11 && x <= 16 && y >= 13 && y <= 15;
}

function isEmptyCorridorTile(x, y) {
  if (y === 9 || y === 10) return x === 12 || x === 15;
  if (y === 11 || y === 17) return x >= 9 && x <= 18;
  if (y === 12 || y === 13 || y === 15 || y === 16 || y === 18 || y === 19) return x === 9 || x === 18;
  if (y === 14) return (x >= 0 && x <= 9) || (x >= 18 && x <= 27);
  return false;
}

function makeWalkableRows(mode) {
  return MAZE_ROWS.map((row, y) => Array.from(row, (ch, x) => {
    if (ch === '.' || ch === 'o' || isEmptyCorridorTile(x, y)) return '1';
    if (mode === 'ghost' && (ch === '-' || isGhostHouseTile(x, y))) return '2';
    return '0';
  }).join(''));
}

const PLAYER_WALKABLE_ROWS = makeWalkableRows('player');
const GHOST_WALKABLE_ROWS = makeWalkableRows('ghost');

function findTiles(predicate, mapRows = MAZE_ROWS) {
  return mapRows.flatMap((row, y) => Array.from(row, (tile, x) => predicate(tile, x, y) ? { x, y } : null).filter(Boolean));
}
const WALL_COORDINATES = findTiles((tile) => tile === '#');
const PELLET_COORDINATES = findTiles((tile) => tile === '.' || tile === 'o').map(({ x, y }) => ({ x, y, type: MAZE_ROWS[y][x] === 'o' ? 'power' : 'regular' }));
const GHOST_ONLY_COORDINATES = findTiles((tile) => tile === '2', GHOST_WALKABLE_ROWS);

const MAZE_META = {
  width: 28,
  height: 31,
  tileSize: 8,
  pelletCount: MAZE_ROWS.reduce((sum, row) => sum + Array.from(row).filter((ch) => ch === '.' || ch === 'o').length, 0),
  playerWalkableTiles: PLAYER_WALKABLE_ROWS.reduce((sum, row) => sum + Array.from(row).filter((ch) => ch === '1').length, 0),
  coordinateSystem: 'zero-based tile coordinates; x grows right, y grows down; (0,0) is the top-left tile',
  legend: { '#': 'wall', '.': 'regular pellet', o: 'power pellet', '-': 'ghost gate (ghosts only)', ' ': 'empty floor or non-playable void; see walkable rows' },
  walkableLegend: { '0': 'blocked / wall / void', '1': 'Pac-Man and ghosts can enter', '2': 'ghost-only house or gate tile' },
  tunnelRows: [14],
  mapNote: 'Classic original arcade maze data: 28x31 tiles, 240 regular pellets and 4 power pellets. Ghost AI in this project is an approximation rather than a cycle-accurate ROM emulation.',
};

function json(res, statusCode, body) {
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  });
  res.end(JSON.stringify(body));
}

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not found');
      return;
    }
    res.writeHead(200, {
      'Content-Type': contentTypes[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
      'X-Content-Type-Options': 'nosniff',
    });
    res.end(data);
  });
}

let latestState = {
  timestamp: new Date().toISOString(),
  frame: 0,
  status: 'waiting-for-game',
  level: 1,
  score: 0,
  lives: 3,
  maze: { width: MAZE_META.width, height: MAZE_META.height, tileSize: MAZE_META.tileSize },
  entities: [],
  pelletsRemaining: null,
  pelletsTotal: null,
  pelletRows: [],
  events: [],
};

const AGENT_ACTIONS = {
  0: { x: 0, y: -1 },
  1: { x: 0, y: 1 },
  2: { x: -1, y: 0 },
  3: { x: 1, y: 0 },
};

function pelletGridFromRows(rows) {
  return rows.map((row) => Array.from(row, (tile) => (tile === '.' || tile === 'o') ? tile : ' '));
}

function countPelletsInGrid(grid) {
  return grid.reduce((sum, row) => sum + row.reduce((inside, tile) => inside + ((tile === '.' || tile === 'o') ? 1 : 0), 0), 0);
}

function cloneAgentState() {
  const pelletGrid = pelletGridFromRows(MAZE_ROWS);
  return {
    initialized: true,
    status: 'ready',
    mode: 'training',
    timestamp: new Date().toISOString(),
    frame: 0,
    level: 1,
    score: 0,
    lives: 3,
    maxSteps: 200,
    stepCount: 0,
    pelletsRemaining: countPelletsInGrid(pelletGrid),
    pelletsTotal: countPelletsInGrid(pelletGrid),
    pelletRows: pelletGrid.map((row) => row.join('')),
    pacman: { id: 'pacman', label: 'Pac-Man', role: 'player', color: '#ffdc3f', x: 13, y: 23, direction: 'left', tile: { x: 13, y: 23 } },
    ghosts: [
      { id: 'blinky', label: 'Blinky', role: 'ghost', color: '#ff4e61', x: 13, y: 14, tile: { x: 13, y: 14 }, direction: 'up', state: 'normal' },
      { id: 'pinky', label: 'Pinky', role: 'ghost', color: '#ff8ad8', x: 12, y: 14, tile: { x: 12, y: 14 }, direction: 'left', state: 'normal' },
      { id: 'inky', label: 'Inky', role: 'ghost', color: '#42e7f4', x: 14, y: 14, tile: { x: 14, y: 14 }, direction: 'down', state: 'normal' },
      { id: 'clyde', label: 'Clyde', role: 'ghost', color: '#ffa45e', x: 15, y: 14, tile: { x: 15, y: 14 }, direction: 'right', state: 'normal' },
    ],
    frightened: false,
    frightenedMsRemaining: 0,
    gameOver: false,
    won: false,
    terminated: false,
    truncated: false,
    lastPellet: null,
    reward: 0,
    eventLog: [],
  };
}

let agentState = cloneAgentState();
let agentControlLock = false;
let agentOwner = null;

function clampTilePosition(x, y) {
  const nextX = Number.isInteger(x) ? x : Math.round(x);
  const nextY = Number.isInteger(y) ? y : Math.round(y);
  if (nextY === 14 && nextX < 0) return { x: MAZE_META.width - 1, y: 14 };
  if (nextY === 14 && nextX >= MAZE_META.width) return { x: 0, y: 14 };
  return { x: nextX, y: nextY };
}

function isWallAt(x, y) {
  if (y < 0 || y >= MAZE_META.height || x < 0 || x >= MAZE_META.width) return true;
  return MAZE_ROWS[y][x] === '#';
}

function isGhostWalkableAt(x, y) {
  if (y < 0 || y >= MAZE_META.height || x < 0 || x >= MAZE_META.width) return false;
  const tile = GHOST_WALKABLE_ROWS[y][x];
  return tile === '1' || tile === '2';
}

function isPelletAt(grid, x, y) {
  if (y < 0 || y >= grid.length || x < 0 || x >= grid[0].length) return false;
  const tile = grid[y][x];
  return tile === '.' || tile === 'o';
}

function normalizeAgentState(state) {
  const pacman = { id: 'pacman', label: 'Pac-Man', role: 'player', color: '#ffdc3f', ...state.pacman, tile: { x: state.pacman.x, y: state.pacman.y }, direction: state.pacman.direction || 'left' };
  const ghostList = Array.isArray(state.ghosts) ? state.ghosts.map((ghost) => ({ ...ghost, tile: { x: ghost.x, y: ghost.y }, direction: ghost.direction || 'up', role: ghost.role || 'ghost' })) : [];
  return {
    ...state,
    pacman,
    ghosts: ghostList,
    maze: { width: MAZE_META.width, height: MAZE_META.height, tileSize: MAZE_META.tileSize },
    entities: [pacman, ...ghostList],
    pelletsRemaining: state.pelletsRemaining,
    pelletsTotal: state.pelletsTotal,
    pelletRows: Array.isArray(state.pelletRows) ? state.pelletRows : [],
    status: state.gameOver ? 'game-over' : state.won ? 'won' : state.terminated ? 'terminated' : 'running',
    timestamp: state.timestamp || new Date().toISOString(),
  };
}

function applyAgentAction(state, action) {
  if (state.terminated || state.gameOver || state.won) {
    return { error: 'Episode already terminated. Call reset() before stepping again.' };
  }

  const before = JSON.parse(JSON.stringify(state));
  const delta = AGENT_ACTIONS[action];
  if (!delta) {
    return { error: 'Action must be one of 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.' };
  }

  const nextX = clampTilePosition(state.pacman.x + delta.x, state.pacman.y + delta.y).x;
  const nextY = clampTilePosition(state.pacman.x + delta.x, state.pacman.y + delta.y).y;
  const currentDirection = { 0: 'up', 1: 'down', 2: 'left', 3: 'right' }[action];

  let reward = 0;
  let moved = false;
  if (!isWallAt(nextX, nextY)) {
    state.pacman.x = nextX;
    state.pacman.y = nextY;
    state.pacman.direction = currentDirection;
    state.pacman.tile = { x: nextX, y: nextY };
    moved = true;
  } else {
    state.pacman.direction = currentDirection;
  }

  const pelletGrid = state.pelletRows.map((row) => row.split(''));
  if (moved && isPelletAt(pelletGrid, state.pacman.x, state.pacman.y)) {
    const tile = pelletGrid[state.pacman.y][state.pacman.x];
    pelletGrid[state.pacman.y][state.pacman.x] = ' ';
    state.pelletRows = pelletGrid.map((row) => row.join(''));
    state.pelletsRemaining = Math.max(0, state.pelletsRemaining - 1);
    state.lastPellet = { x: state.pacman.x, y: state.pacman.y, type: tile === 'o' ? 'power' : 'regular' };
    const pelletReward = tile === 'o' ? 3 : 1;
    state.score += pelletReward;
    state.reward += pelletReward;
    reward += pelletReward;
    if (tile === 'o') {
      state.frightened = true;
      state.frightenedMsRemaining = 60;
    }
  }

  state.stepCount += 1;
  state.frame += 1;
  state.status = 'running';
  state.timestamp = new Date().toISOString();

  state.ghosts = state.ghosts.map((ghost, index) => {
    const shift = index % 2 === 0 ? 1 : -1;
    const nextGhost = clampTilePosition(
      ghost.x + (ghost.direction === 'left' ? -1 : ghost.direction === 'right' ? 1 : 0),
      ghost.y + (ghost.direction === 'up' ? -1 : ghost.direction === 'down' ? 1 : 0),
    );
    const candidate = clampTilePosition(
      ghost.x + shift * (Math.abs(state.pacman.x - ghost.x) >= Math.abs(state.pacman.y - ghost.y) ? (state.pacman.x >= ghost.x ? 1 : -1) : 0),
      ghost.y + shift * (Math.abs(state.pacman.y - ghost.y) > Math.abs(state.pacman.x - ghost.x) ? (state.pacman.y >= ghost.y ? 1 : -1) : 0),
    );
    const chosen = isGhostWalkableAt(candidate.x, candidate.y)
      ? candidate
      : isGhostWalkableAt(nextGhost.x, nextGhost.y)
        ? nextGhost
        : { x: ghost.x, y: ghost.y };
    return {
      ...ghost,
      x: chosen.x,
      y: chosen.y,
      tile: { x: chosen.x, y: chosen.y },
      direction: ghost.direction,
      state: state.frightened ? 'frightened' : 'normal',
    };
  });

  if (state.frightenedMsRemaining > 0) {
    state.frightenedMsRemaining -= 1;
    if (state.frightenedMsRemaining <= 0) state.frightened = false;
  }

  const ghostCaught = state.ghosts.some((ghost) => ghost.x === state.pacman.x && ghost.y === state.pacman.y);
  if (ghostCaught) {
    if (state.frightened) {
      const ghostReward = 5;
      state.score += ghostReward;
      reward += ghostReward;
      state.reward += ghostReward;
      state.ghosts = state.ghosts.map((ghost) => ghost.x === state.pacman.x && ghost.y === state.pacman.y ? { ...ghost, x: ghost.id === 'blinky' ? 13 : ghost.id === 'pinky' ? 12 : ghost.id === 'inky' ? 14 : 15, y: 14, tile: { x: ghost.id === 'blinky' ? 13 : ghost.id === 'pinky' ? 12 : ghost.id === 'inky' ? 14 : 15, y: 14 }, state: 'eyes' } : ghost);
      state.eventLog.push('ghost-captured');
    } else {
      const lifePenalty = -10;
      state.lives -= 1;
      state.score += lifePenalty;
      reward += lifePenalty;
      state.reward += lifePenalty;
      state.eventLog.push('life-lost');
      if (state.lives <= 0) {
        state.gameOver = true;
        state.terminated = true;
        const failPenalty = -20;
        state.score += failPenalty;
        state.reward += failPenalty;
        reward += failPenalty;
        state.status = 'game-over';
      } else {
        state.pacman = { id: 'pacman', label: 'Pac-Man', role: 'player', color: '#ffdc3f', x: 13, y: 23, direction: 'left', tile: { x: 13, y: 23 } };
        state.ghosts = [
          { id: 'blinky', label: 'Blinky', role: 'ghost', color: '#ff4e61', x: 13, y: 14, tile: { x: 13, y: 14 }, direction: 'up', state: 'normal' },
          { id: 'pinky', label: 'Pinky', role: 'ghost', color: '#ff8ad8', x: 12, y: 14, tile: { x: 12, y: 14 }, direction: 'left', state: 'normal' },
          { id: 'inky', label: 'Inky', role: 'ghost', color: '#42e7f4', x: 14, y: 14, tile: { x: 14, y: 14 }, direction: 'down', state: 'normal' },
          { id: 'clyde', label: 'Clyde', role: 'ghost', color: '#ffa45e', x: 15, y: 14, tile: { x: 15, y: 14 }, direction: 'right', state: 'normal' },
        ];
      }
    }
  }

  if (state.pelletsRemaining === 0 && !state.won) {
    state.won = true;
    state.terminated = true;
    const winReward = 50;
    state.score += winReward;
    reward += winReward;
    state.reward += winReward;
    state.status = 'won';
  }

  if (state.stepCount >= state.maxSteps && !state.terminated) {
    state.truncated = true;
    state.terminated = false;
  }

  if (state.terminated || state.gameOver || state.won) {
    state.status = state.gameOver ? 'game-over' : state.won ? 'won' : 'terminated';
  }

  const result = normalizeAgentState(state);
  return {
    state: result,
    reward,
    terminated: result.terminated,
    truncated: result.truncated,
    info: {
      stepCount: result.stepCount,
      action,
      moved,
      score: result.score,
      lives: result.lives,
      won: result.won,
      gameOver: result.gameOver,
      lastPellet: result.lastPellet,
      eventLog: result.eventLog.slice(-10),
    },
  };
}

const stateHistory = [];
const eventHistory = [];
const sseClients = new Set();
let stateSequence = 0;

function publishState(state) {
  stateSequence += 1;
  latestState = { ...state, serverSequence: stateSequence };
  stateHistory.push({
    timestamp: latestState.timestamp || new Date().toISOString(),
    frame: latestState.frame,
    status: latestState.status,
    score: latestState.score,
    lives: latestState.lives,
    pelletsRemaining: latestState.pelletsRemaining,
    entities: (latestState.entities || []).map((entity) => ({
      id: entity.id,
      x: entity.x,
      y: entity.y,
      tile: entity.tile,
      direction: entity.direction,
      state: entity.state,
    })),
  });
  if (stateHistory.length > HISTORY_LIMIT) stateHistory.shift();

  for (const event of (state.newEvents || [])) {
    if (!event || !event.id) continue;
    if (eventHistory.some((existing) => existing.id === event.id)) continue;
    eventHistory.push(event);
    if (eventHistory.length > EVENT_LIMIT) eventHistory.shift();
  }
  latestState.events = eventHistory.slice(-20);
  latestState.newEvents = undefined;

  const message = `event: state\ndata: ${JSON.stringify(latestState)}\n\n`;
  for (const client of sseClients) {
    try { client.write(message); } catch { sseClients.delete(client); }
  }
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > MAX_BODY) {
        reject(new Error('Payload too large'));
        req.destroy();
      }
    });
    req.on('end', () => {
      try { resolve(JSON.parse(body || '{}')); }
      catch { reject(new Error('Invalid JSON body')); }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const parsed = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const pathname = decodeURIComponent(parsed.pathname);

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    });
    res.end();
    return;
  }

  if (req.method === 'GET' && pathname === '/api/health') {
    json(res, 200, { ok: true, service: 'pacman-live-tracker', uptimeSeconds: Math.floor(process.uptime()), connectedStreams: sseClients.size, stateSequence, lastUpdate: latestState.timestamp });
    return;
  }
  if (req.method === 'GET' && pathname === '/api/maze') {
    json(res, 200, { ...MAZE_META, rows: MAZE_ROWS, playerWalkableRows: PLAYER_WALKABLE_ROWS, ghostWalkableRows: GHOST_WALKABLE_ROWS, wallCoordinates: WALL_COORDINATES, pelletCoordinates: PELLET_COORDINATES, ghostOnlyCoordinates: GHOST_ONLY_COORDINATES, pacmanStart: { x: 13, y: 23 }, ghostStarts: [
      { id: 'blinky', x: 13, y: 14 }, { id: 'pinky', x: 12, y: 14 }, { id: 'inky', x: 14, y: 14 }, { id: 'clyde', x: 15, y: 14 },
    ] });
    return;
  }
  if (req.method === 'POST' && pathname === '/api/agent/reset') {
    try {
      const payload = await readJson(req).catch(() => ({}));
      if (agentOwner && payload.owner_id !== agentOwner) {
        json(res, 409, { error: 'Agent state is exclusively owned by another session.' });
        return;
      }
      const requestedMaxSteps = Number(payload.max_steps ?? payload.maxSteps ?? 200);
      agentState = cloneAgentState();
      if (Number.isFinite(requestedMaxSteps) && requestedMaxSteps > 0) agentState.maxSteps = requestedMaxSteps;
      agentState.status = 'ready';
      agentState.initialized = true;
      const state = normalizeAgentState(agentState);
      json(res, 200, { ok: true, state, reward: 0, terminated: false, truncated: false, info: { maxSteps: agentState.maxSteps, reset: true } });
    } catch (error) {
      json(res, 400, { error: error.message || 'Reset failed' });
    }
    return;
  }
  if (req.method === 'POST' && pathname === '/api/agent/step') {
    if (agentControlLock) {
      json(res, 409, { error: 'A training step is already in progress.' });
      return;
    }
    try {
      const payload = await readJson(req).catch(() => ({}));
      if (agentOwner && payload.owner_id !== agentOwner) {
        json(res, 409, { error: 'Agent state is exclusively owned by another session.' });
        return;
      }
      agentControlLock = true;
      const action = Number(payload.action ?? payload.step ?? payload.actionIndex);
      if (!Number.isInteger(action) || action < 0 || action > 3) {
        json(res, 400, { error: 'Action must be an integer between 0 and 3. Mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.' });
        return;
      }
      const result = applyAgentAction(agentState, action);
      if (result && result.error) {
        json(res, 409, { error: result.error });
        return;
      }
      json(res, 200, result);
    } catch (error) {
      json(res, error.message === 'Payload too large' ? 413 : 400, { error: error.message || 'Could not read step payload' });
    } finally {
      agentControlLock = false;
    }
    return;
  }
  if (req.method === 'POST' && pathname === '/api/agent/claim') {
    try {
      const payload = await readJson(req);
      const ownerId = payload.owner_id;
      if (typeof ownerId !== 'string' || ownerId.length < 1 || ownerId.length > 128) {
        json(res, 400, { error: 'owner_id must be a non-empty string of at most 128 characters.' });
        return;
      }
      if (agentControlLock || (agentOwner && agentOwner !== ownerId)) {
        json(res, 409, { error: 'Agent state is already in use by another session.' });
        return;
      }
      agentOwner = ownerId;
      json(res, 200, { claimed: true });
    } catch (error) {
      json(res, error.message === 'Payload too large' ? 413 : 400, { error: error.message || 'Could not read claim payload' });
    }
    return;
  }
  if (req.method === 'POST' && pathname === '/api/agent/release') {
    try {
      const payload = await readJson(req);
      if (agentControlLock) {
        json(res, 409, { error: 'An agent step is still in progress.' });
        return;
      }
      if (!agentOwner || payload.owner_id !== agentOwner) {
        json(res, 409, { error: 'This session does not own the agent state.' });
        return;
      }
      agentOwner = null;
      json(res, 200, { released: true });
    } catch (error) {
      json(res, error.message === 'Payload too large' ? 413 : 400, { error: error.message || 'Could not read release payload' });
    }
    return;
  }
  if (req.method === 'GET' && pathname === '/api/agent/state') {
    if (!agentState || !agentState.initialized) {
      json(res, 503, { error: 'Agent training state is not initialized. Call /api/agent/reset first.' });
      return;
    }
    json(res, 200, normalizeAgentState(agentState));
    return;
  }
  if (req.method === 'GET' && pathname === '/api/state') {
    json(res, 200, latestState);
    return;
  }
  if (req.method === 'GET' && pathname === '/api/history') {
    const limit = Math.min(240, Math.max(1, Number(parsed.searchParams.get('limit') || 120)));
    json(res, 200, { count: Math.min(limit, stateHistory.length), samples: stateHistory.slice(-limit) });
    return;
  }
  if (req.method === 'GET' && pathname === '/api/events') {
    json(res, 200, { count: eventHistory.length, events: eventHistory });
    return;
  }
  if (req.method === 'GET' && pathname === '/api/stream') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'X-Accel-Buffering': 'no',
    });
    res.write(`event: state\ndata: ${JSON.stringify(latestState)}\n\n`);
    sseClients.add(res);
    const heartbeat = setInterval(() => {
      if (!res.writableEnded) res.write(`: heartbeat ${Date.now()}\n\n`);
    }, 15000);
    req.on('close', () => { clearInterval(heartbeat); sseClients.delete(res); });
    return;
  }
  if (req.method === 'POST' && pathname === '/api/state') {
    try {
      const state = await readJson(req);
      if (!state || !Array.isArray(state.entities) || !state.maze || state.maze.width !== 28 || state.maze.height !== 31) {
        json(res, 400, { error: 'Expected a game state with entities and a 28x31 maze descriptor.' });
        return;
      }
      publishState(state);
      json(res, 202, { accepted: true, serverSequence: stateSequence, timestamp: latestState.timestamp });
    } catch (error) {
      json(res, error.message === 'Payload too large' ? 413 : 400, { error: error.message || 'Could not read request body' });
    }
    return;
  }

  if (req.method === 'GET') {
    const safePath = pathname === '/' ? '/index.html' : pathname;
    const target = path.resolve(PUBLIC_DIR, `.${safePath}`);
    if (!target.startsWith(PUBLIC_DIR + path.sep) && target !== path.join(PUBLIC_DIR, 'index.html')) {
      res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Forbidden');
      return;
    }
    sendFile(res, target);
    return;
  }

  json(res, 405, { error: 'Method not allowed' });
});

server.listen(PORT, HOST, () => {
  console.log(`Pac-Man Live Tracker running at http://localhost:${PORT}`);
  console.log(`API: /api/maze  /api/state  /api/stream  /api/history  /api/events`);
});

function shutdown() {
  for (const client of sseClients) client.end();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1000).unref();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
