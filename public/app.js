'use strict';

(() => {
  const WIDTH = 28;
  const HEIGHT = 31;
  const CELL = 20;
  const SOURCE_TILE = 8;
  const BOARD_W = WIDTH * CELL;
  const BOARD_H = HEIGHT * CELL;
  const DIRECTIONS = {
    up: { x: 0, y: -1, angle: -Math.PI / 2 },
    right: { x: 1, y: 0, angle: 0 },
    down: { x: 0, y: 1, angle: Math.PI / 2 },
    left: { x: -1, y: 0, angle: Math.PI },
  };
  const DIRECTION_ORDER = ['up', 'left', 'down', 'right'];
  const GHOST_CONFIG = [
    { id: 'blinky', label: 'Blinky', color: '#ff4e61', home: { x: 13, y: 14 }, speed: 4.2, bias: 0 },
    { id: 'pinky', label: 'Pinky', color: '#ff8ad8', home: { x: 12, y: 14 }, speed: 4.0, bias: 1 },
    { id: 'inky', label: 'Inky', color: '#42e7f4', home: { x: 14, y: 14 }, speed: 3.9, bias: 2 },
    { id: 'clyde', label: 'Clyde', color: '#ffa45e', home: { x: 15, y: 14 }, speed: 3.8, bias: 3 },
  ];
  const $ = (id) => document.getElementById(id);
  const canvas = $('gameCanvas');
  const ctx = canvas.getContext('2d');

  let mazeRows = [];
  let playerWalkableRows = [];
  let ghostWalkableRows = [];
  let pelletGrid = [];
  let pelletsTotal = 0;
  let pelletsRemaining = 0;
  let pacman;
  let ghosts = [];
  let score = 0;
  let lives = 3;
  let level = 1;
  let frameCount = 0;
  let wallClockMs = 0;
  let gameTimeMs = 0;
  let lastFrameAt = 0;
  let lastUiAt = 0;
  let lastApiAt = -1000;
  let lastPelletAt = null;
  let frightenedUntilMs = 0;
  let invulnerableUntilMs = 0;
  let paused = false;
  let gameOver = false;
  let levelClearAt = 0;
  let showGrid = false;
  let showTrails = false;
  let eventCounter = 0;
  let sessionId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
  let events = [];
  let eventOutbox = [];
  let apiSendPending = false;
  let apiOnline = false;
  let apiSuccessTimes = [];
  let lastSnapshot = null;
  let eventDirty = true;
  let entityDirty = true;
  let animationFrame = null;

  const trails = new Map();

  function setCanvasScale() {
    const dpr = Math.max(1, Math.min(2.2, window.devicePixelRatio || 1));
    canvas.width = Math.round(BOARD_W * dpr);
    canvas.height = Math.round(BOARD_H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawBoard();
  }

  function isGhostHouseTile(x, y) {
    return x >= 11 && x <= 16 && y >= 13 && y <= 15;
  }

  function canEnter(x, y, type) {
    if (y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH) return false;
    const mask = type === 'ghost' ? ghostWalkableRows : playerWalkableRows;
    const value = mask[y]?.[x] || '0';
    return value === '1' || (type === 'ghost' && value === '2');
  }

  function normalizeCell(x, y) {
    if (y === 14 && x < 0) x = WIDTH - 1;
    if (y === 14 && x >= WIDTH) x = 0;
    return { x, y };
  }

  function destination(x, y, direction) {
    const d = DIRECTIONS[direction];
    if (!d) return null;
    return normalizeCell(x + d.x, y + d.y);
  }

  function canMove(x, y, direction, type) {
    const next = destination(x, y, direction);
    return !!next && canEnter(next.x, next.y, type);
  }

  function initialPelletGrid() {
    return mazeRows.map((row) => Array.from(row, (ch) => (ch === '.' || ch === 'o') ? ch : ' '));
  }

  function countPellets() {
    return pelletGrid.reduce((sum, row) => sum + row.reduce((inner, value) => inner + ((value === '.' || value === 'o') ? 1 : 0), 0), 0);
  }

  function makeActor({ id, label, role, color, x, y, speed, direction = 'left', bias = 0 }) {
    return {
      id, label, role, color, x, y, tileX: x, tileY: y,
      fromX: x, fromY: y, target: null, progress: 0,
      direction, wantedDirection: direction, speed, bias,
      state: 'normal', returning: false, mouthPhase: 0,
    };
  }

  function resetActors() {
    pacman = makeActor({ id: 'pacman', label: 'Pac-Man', role: 'player', color: '#ffdc3f', x: 13, y: 23, speed: Math.min(7.2, 6.8 + (level - 1) * 0.12), direction: 'left' });
    ghosts = GHOST_CONFIG.map((config) => makeActor({ ...config, role: 'ghost', x: config.home.x, y: config.home.y, direction: 'up' }));
    for (const ghost of ghosts) trails.set(ghost.id, []);
    trails.set('pacman', []);
    invulnerableUntilMs = gameTimeMs + 1700;
  }

  function eventTime() {
    const d = new Date();
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
  }

  function addEvent(type, title, detail, kind = 'normal', coords = null) {
    eventCounter += 1;
    const event = {
      id: `${sessionId}-${eventCounter}`,
      timestamp: new Date().toISOString(),
      clock: eventTime(),
      elapsedMs: Math.round(gameTimeMs),
      type, title, detail, kind,
      ...(coords ? { coords: { x: coords.x, y: coords.y } } : {}),
    };
    events.push(event);
    if (events.length > 35) events.shift();
    eventOutbox.push(event);
    eventDirty = true;
    return event;
  }

  function collectPelletAt(x, y, initial = false) {
    if (y < 0 || y >= HEIGHT || x < 0 || x >= WIDTH) return;
    const value = pelletGrid[y]?.[x];
    if (value !== '.' && value !== 'o') return;
    pelletGrid[y][x] = ' ';
    pelletsRemaining = Math.max(0, pelletsRemaining - 1);
    score += value === 'o' ? 50 : 10;
    lastPelletAt = { x, y, type: value === 'o' ? 'power' : 'regular' };
    if (value === 'o') {
      frightenedUntilMs = gameTimeMs + Math.max(4500, 8500 - (level - 1) * 450);
      ghosts.forEach((ghost) => { if (!ghost.returning) ghost.state = 'frightened'; });
      addEvent('power-pellet', 'POWER PELLET', `Frightened mode active · tile (${x}, ${y})`, 'power', { x, y });
    } else if (!initial) {
      addEvent('pellet', 'PELLET EATEN', `+10 points · tile (${x}, ${y})`, 'normal', { x, y });
    }
    if (pelletsRemaining === 0) {
      addEvent('level-clear', 'MAZE CLEARED', `Level ${level} complete`, 'success', { x, y });
      levelClearAt = gameTimeMs + 1700;
    }
  }

  function initialState() {
    pelletGrid = initialPelletGrid();
    pelletsTotal = countPellets();
    pelletsRemaining = pelletsTotal;
    score = 0;
    lives = 3;
    level = 1;
    frameCount = 0;
    wallClockMs = 0;
    gameTimeMs = 0;
    frightenedUntilMs = 0;
    paused = false;
    gameOver = false;
    levelClearAt = 0;
    lastPelletAt = null;
    events = [];
    eventOutbox = [];
    eventCounter = 0;
    sessionId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
    resetActors();
    addEvent('start', 'RUN STARTED', 'Classic maze loaded · coordinate stream active', 'success', { x: pacman.tileX, y: pacman.tileY });
    syncOverlay();
    updateDashboard(true);
  }

  function requestNewLevel() {
    level += 1;
    pelletGrid = initialPelletGrid();
    pelletsTotal = countPellets();
    pelletsRemaining = pelletsTotal;
    frightenedUntilMs = 0;
    gameOver = false;
    levelClearAt = 0;
    resetActors();
    addEvent('level', 'LEVEL ADVANCED', `Level ${String(level).padStart(2, '0')} · ghost speed increased`, 'success', { x: pacman.tileX, y: pacman.tileY });
    syncOverlay();
  }

  function actorStepTarget(actor, direction) {
    const next = destination(actor.tileX, actor.tileY, direction);
    if (!next || !canEnter(next.x, next.y, actor.role === 'ghost' ? 'ghost' : 'player')) return null;
    const isWarp = next.y === 14 && Math.abs(next.x - actor.tileX) > 1;
    return { ...next, warp: isWarp };
  }

  function nextPlayerDirection() {
    if (pacman.wantedDirection && canMove(pacman.tileX, pacman.tileY, pacman.wantedDirection, 'player')) return pacman.wantedDirection;
    if (pacman.direction && canMove(pacman.tileX, pacman.tileY, pacman.direction, 'player')) return pacman.direction;
    return null;
  }

  function targetForGhost(ghost) {
    const px = Math.max(0, Math.min(WIDTH - 1, Math.round(pacman.x)));
    const py = Math.max(0, Math.min(HEIGHT - 1, Math.round(pacman.y)));
    if (ghost.returning) return GHOST_CONFIG.find((item) => item.id === ghost.id).home;
    if (ghost.id === 'pinky') {
      const d = DIRECTIONS[pacman.direction] || DIRECTIONS.left;
      const target = normalizeCell(px + d.x * 3, py + d.y * 3);
      if (canEnter(target.x, target.y, 'ghost')) return target;
    }
    if (ghost.id === 'inky') {
      const d = DIRECTIONS[pacman.direction] || DIRECTIONS.left;
      const red = ghosts.find((item) => item.id === 'blinky');
      const ahead = normalizeCell(px + d.x * 2, py + d.y * 2);
      const target = normalizeCell(ahead.x + (ahead.x - Math.round(red.x)), ahead.y + (ahead.y - Math.round(red.y)));
      if (canEnter(target.x, target.y, 'ghost')) return target;
    }
    if (ghost.id === 'clyde') {
      const distance = Math.abs(ghost.tileX - px) + Math.abs(ghost.tileY - py);
      if (distance < 8) return { x: 1, y: 29 };
    }
    return { x: px, y: py };
  }

  function neighborFor(x, y, direction) {
    return destination(x, y, direction);
  }

  function ghostPathDirection(ghost, target) {
    const start = { x: ghost.tileX, y: ghost.tileY };
    if (start.x === target.x && start.y === target.y) return null;
    const queue = [{ ...start, first: null }];
    const visited = new Set([`${start.x},${start.y}`]);
    const ordered = DIRECTION_ORDER.slice(ghost.bias).concat(DIRECTION_ORDER.slice(0, ghost.bias));
    let readIndex = 0;
    while (readIndex < queue.length && readIndex < 900) {
      const node = queue[readIndex++];
      for (const direction of ordered) {
        const next = neighborFor(node.x, node.y, direction);
        if (!next || !canEnter(next.x, next.y, 'ghost')) continue;
        const key = `${next.x},${next.y}`;
        if (visited.has(key)) continue;
        visited.add(key);
        const first = node.first || direction;
        if (next.x === target.x && next.y === target.y) return first;
        queue.push({ ...next, first });
      }
    }
    return null;
  }

  function chooseGhostDirection(ghost) {
    const moves = DIRECTION_ORDER.filter((direction) => canMove(ghost.tileX, ghost.tileY, direction, 'ghost'));
    if (!moves.length) return null;
    if (ghost.returning) {
      const home = GHOST_CONFIG.find((item) => item.id === ghost.id).home;
      if (ghost.tileX === home.x && ghost.tileY === home.y) {
        ghost.returning = false;
        ghost.state = (frightenedUntilMs > gameTimeMs) ? 'frightened' : 'normal';
        return null;
      }
      return ghostPathDirection(ghost, home) || moves[0];
    }
    if (frightenedUntilMs > gameTimeMs) {
      const reverse = ghost.direction ? ({ up: 'down', down: 'up', left: 'right', right: 'left' })[ghost.direction] : null;
      const candidates = moves.length > 1 ? moves.filter((direction) => direction !== reverse) : moves;
      const target = { x: Math.round(pacman.x), y: Math.round(pacman.y) };
      candidates.sort((a, b) => {
        const ta = destination(ghost.tileX, ghost.tileY, a);
        const tb = destination(ghost.tileX, ghost.tileY, b);
        const da = Math.abs(ta.x - target.x) + Math.abs(ta.y - target.y) + Math.random() * 0.7;
        const db = Math.abs(tb.x - target.x) + Math.abs(tb.y - target.y) + Math.random() * 0.7;
        return db - da;
      });
      return candidates[0];
    }
    const direction = ghostPathDirection(ghost, targetForGhost(ghost));
    return direction || (moves.includes(ghost.direction) ? ghost.direction : moves[0]);
  }

  function startActorMove(actor) {
    let direction;
    if (actor.role === 'player') {
      direction = nextPlayerDirection();
    } else {
      direction = chooseGhostDirection(actor);
    }
    if (!direction) return false;
    const target = actorStepTarget(actor, direction);
    if (!target) return false;
    if (actor.role === 'player') collectPelletAt(actor.tileX, actor.tileY);
    actor.direction = direction;
    actor.target = target;
    actor.fromX = actor.tileX;
    actor.fromY = actor.tileY;
    actor.progress = 0;
    return true;
  }

  function updateEntity(actor, dtSeconds) {
    let remaining = dtSeconds;
    let safety = 0;
    while (remaining > 0.00001 && safety++ < 4) {
      if (!actor.target) {
        if (!startActorMove(actor)) break;
      }
      let speed = actor.speed;
      if (actor.role === 'ghost' && actor.returning) speed = 7.2;
      else if (actor.role === 'ghost' && frightenedUntilMs > gameTimeMs) speed *= 0.73;
      const need = (1 - actor.progress) / Math.max(0.01, speed);
      const used = Math.min(remaining, need);
      actor.progress = Math.min(1, actor.progress + used * speed);
      const t = actor.progress;
      if (actor.target.warp) {
        const leavingLeft = actor.fromX === 0 && actor.target.x === WIDTH - 1;
        const leavingRight = actor.fromX === WIDTH - 1 && actor.target.x === 0;
        if (t < 0.5) {
          actor.x = leavingLeft ? 0 : WIDTH - 1;
          actor.y = actor.fromY;
        } else {
          actor.x = leavingLeft ? WIDTH - 1 : 0;
          actor.y = actor.fromY;
        }
      } else {
        actor.x = actor.fromX + (actor.target.x - actor.fromX) * t;
        actor.y = actor.fromY + (actor.target.y - actor.fromY) * t;
      }
      remaining -= used;
      if (actor.progress >= 0.99999) {
        actor.tileX = actor.target.x;
        actor.tileY = actor.target.y;
        actor.x = actor.tileX;
        actor.y = actor.tileY;
        actor.target = null;
        actor.progress = 0;
        if (actor.role === 'player') collectPelletAt(actor.tileX, actor.tileY);
        entityDirty = true;
      } else {
        break;
      }
    }
  }

  function recordTrails() {
    const actors = [pacman, ...ghosts];
    for (const actor of actors) {
      const points = trails.get(actor.id) || [];
      const last = points[points.length - 1];
      if (!last || Math.abs(last.x - actor.x) + Math.abs(last.y - actor.y) > 0.055) {
        points.push({ x: actor.x, y: actor.y });
        if (points.length > 75) points.shift();
      }
      trails.set(actor.id, points);
    }
  }

  function isCollision(a, b) {
    const dx = Math.abs(a.x - b.x);
    const dy = Math.abs(a.y - b.y);
    // The tunnel has a discontinuity in screen coordinates; those edge crossings should not collide across the board.
    if (dx > 2) return false;
    return Math.hypot(dx, dy) < 0.5;
  }

  function checkCollisions() {
    if (gameTimeMs < invulnerableUntilMs || gameOver || paused || levelClearAt > 0) return;
    for (const ghost of ghosts) {
      if (ghost.returning || !isCollision(pacman, ghost)) continue;
      if (frightenedUntilMs > gameTimeMs) {
        score += 200;
        ghost.returning = true;
        ghost.state = 'eyes';
        addEvent('ghost-eaten', 'GHOST CAPTURED', `${ghost.label} · +200 points`, 'success', { x: ghost.tileX, y: ghost.tileY });
      } else {
        lives -= 1;
        addEvent('collision', 'PLAYER HIT', `${ghost.label} collision · ${lives} ${lives === 1 ? 'life' : 'lives'} remaining`, 'danger', { x: Math.round(pacman.x), y: Math.round(pacman.y) });
        if (lives <= 0) {
          gameOver = true;
          paused = false;
          addEvent('game-over', 'GAME OVER', `Final score ${score}`, 'danger');
        } else {
          resetActors();
          invulnerableUntilMs = gameTimeMs + 2200;
        }
        syncOverlay();
      }
      break;
    }
  }

  function updateGame(dtSeconds) {
    updateEntity(pacman, dtSeconds);
    for (const ghost of ghosts) updateEntity(ghost, dtSeconds);
    if (frightenedUntilMs > 0 && gameTimeMs >= frightenedUntilMs) {
      frightenedUntilMs = 0;
      ghosts.forEach((ghost) => { if (!ghost.returning) ghost.state = 'normal'; });
      addEvent('power-end', 'POWER MODE ENDED', 'Ghost movement returned to chase mode', 'normal');
    }
    checkCollisions();
    if (levelClearAt && gameTimeMs >= levelClearAt) requestNewLevel();
  }

  function drawWall(x, y) {
    const px = x * CELL;
    const py = y * CELL;
    ctx.fillStyle = '#07143f';
    ctx.strokeStyle = '#3472ff';
    ctx.lineWidth = 1.15;
    ctx.beginPath();
    ctx.roundRect(px + 1.2, py + 1.2, CELL - 2.4, CELL - 2.4, 4.2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = '#5c87ff55';
    ctx.lineWidth = 0.65;
    ctx.beginPath();
    ctx.roundRect(px + 3, py + 3, CELL - 6, CELL - 6, 3);
    ctx.stroke();
  }

  function drawPellet(x, y, type) {
    const px = x * CELL + CELL / 2;
    const py = y * CELL + CELL / 2;
    if (type === '.') {
      ctx.fillStyle = '#e7ecfa';
      ctx.beginPath();
      ctx.arc(px, py, 1.65, 0, Math.PI * 2);
      ctx.fill();
    } else if (type === 'o') {
      ctx.fillStyle = '#ffe58b';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#ffe58b88';
      ctx.beginPath();
      ctx.arc(px, py, 5.1 + Math.sin(gameTimeMs / 140) * 0.45, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }

  function drawCoordinateGrid() {
    ctx.save();
    ctx.strokeStyle = '#9cc2ff1d';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    for (let x = 0; x <= WIDTH; x++) { ctx.moveTo(x * CELL + 0.5, 0); ctx.lineTo(x * CELL + 0.5, BOARD_H); }
    for (let y = 0; y <= HEIGHT; y++) { ctx.moveTo(0, y * CELL + 0.5); ctx.lineTo(BOARD_W, y * CELL + 0.5); }
    ctx.stroke();
    ctx.fillStyle = '#7f9dc8aa';
    ctx.font = '7px monospace';
    ctx.textAlign = 'center';
    for (let x = 0; x < WIDTH; x += 2) ctx.fillText(String(x), x * CELL + CELL / 2, 8);
    ctx.textAlign = 'left';
    for (let y = 0; y < HEIGHT; y += 2) ctx.fillText(String(y), 2, y * CELL + 8);
    ctx.restore();
  }

  function drawTrails() {
    if (!showTrails) return;
    for (const actor of [pacman, ...ghosts]) {
      const points = trails.get(actor.id) || [];
      if (points.length < 2) continue;
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(points[0].x * CELL + CELL / 2, points[0].y * CELL + CELL / 2);
      for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x * CELL + CELL / 2, points[i].y * CELL + CELL / 2);
      ctx.strokeStyle = `${actor.color}45`;
      ctx.lineWidth = actor.id === 'pacman' ? 2 : 1.3;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.restore();
    }
  }

  function drawPacman(actor) {
    const cx = actor.x * CELL + CELL / 2;
    const cy = actor.y * CELL + CELL / 2;
    const radius = 8.2;
    const dir = DIRECTIONS[actor.direction] || DIRECTIONS.left;
    const mouth = 0.1 + Math.abs(Math.sin(gameTimeMs / 85)) * 0.42;
    ctx.save();
    if (gameTimeMs < invulnerableUntilMs && Math.floor(gameTimeMs / 110) % 2 === 0) ctx.globalAlpha = 0.5;
    ctx.fillStyle = '#ffdf3c';
    ctx.shadowColor = '#ffd62d6e';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, dir.angle + mouth, dir.angle + Math.PI * 2 - mouth, false);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function drawGhost(actor) {
    const cx = actor.x * CELL + CELL / 2;
    const cy = actor.y * CELL + CELL / 2;
    const r = 8.2;
    const color = actor.state === 'frightened' && !actor.returning
      ? ((frightenedUntilMs - gameTimeMs < 1800 && Math.floor(gameTimeMs / 120) % 2 === 0) ? '#f2f5ff' : '#3166ff')
      : actor.returning ? '#111a2f' : actor.color;
    ctx.save();
    ctx.fillStyle = color;
    ctx.shadowColor = actor.state === 'frightened' ? '#356dff88' : `${color}50`;
    ctx.shadowBlur = 7;
    ctx.beginPath();
    ctx.moveTo(cx - r, cy + r * 0.78);
    ctx.lineTo(cx - r, cy - 0.8);
    ctx.arc(cx, cy - 0.8, r, Math.PI, 0, false);
    ctx.lineTo(cx + r, cy + r * 0.78);
    ctx.lineTo(cx + r * 0.52, cy + r * 0.43);
    ctx.lineTo(cx + r * 0.08, cy + r * 0.78);
    ctx.lineTo(cx - r * 0.36, cy + r * 0.43);
    ctx.lineTo(cx - r, cy + r * 0.78);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;
    if (actor.state !== 'frightened' && !actor.returning) {
      const dir = DIRECTIONS[actor.direction] || DIRECTIONS.left;
      const eyeOffsetX = dir.x * 1.25;
      const eyeOffsetY = dir.y * 1.25;
      for (const side of [-1, 1]) {
        ctx.fillStyle = '#f7fbff';
        ctx.beginPath();
        ctx.ellipse(cx + side * 3.1, cy - 1.25, 2.6, 3.2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#13203e';
        ctx.beginPath();
        ctx.arc(cx + side * 3.1 + eyeOffsetX, cy - 1.2 + eyeOffsetY, 1.25, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (actor.state === 'frightened' && !actor.returning) {
      ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(cx - 2.7, cy - 1, 1, 0, Math.PI * 2); ctx.arc(cx + 2.7, cy - 1, 1, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(cx - 4, cy + 3); ctx.lineTo(cx - 2, cy + 1); ctx.lineTo(cx, cy + 3); ctx.lineTo(cx + 2, cy + 1); ctx.lineTo(cx + 4, cy + 3); ctx.stroke();
    }
    ctx.restore();
  }

  function drawBoard() {
    if (!mazeRows.length) return;
    ctx.clearRect(0, 0, BOARD_W, BOARD_H);
    ctx.fillStyle = '#040817';
    ctx.fillRect(0, 0, BOARD_W, BOARD_H);
    // The maze is fixed; only the pellet layer and actors change during play.
    for (let y = 0; y < HEIGHT; y++) {
      for (let x = 0; x < WIDTH; x++) {
        const ch = mazeRows[y][x];
        if (ch === '#') drawWall(x, y);
        else if (ch === '-') {
          ctx.fillStyle = '#15203d';
          ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
          ctx.strokeStyle = '#ffbfff';
          ctx.lineWidth = 2.5;
          ctx.beginPath(); ctx.moveTo(x * CELL + 2, y * CELL + CELL / 2); ctx.lineTo(x * CELL + CELL - 2, y * CELL + CELL / 2); ctx.stroke();
        } else if (isGhostHouseTile(x, y)) {
          ctx.fillStyle = '#090f22';
          ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
        }
        const pellet = pelletGrid[y]?.[x];
        if (pellet === '.' || pellet === 'o') drawPellet(x, y, pellet);
      }
    }
    drawTrails();
    if (pacman) drawPacman(pacman);
    for (const ghost of ghosts) drawGhost(ghost);
    if (showGrid) drawCoordinateGrid();
  }

  function syncOverlay() {
    const overlay = $('startOverlay');
    if (gameOver) {
      overlay.hidden = false;
      $('overlayTitle').textContent = 'GAME OVER';
      $('overlaySub').textContent = 'Press R or RESTART for another run.';
    } else if (paused) {
      overlay.hidden = false;
      $('overlayTitle').textContent = 'PAUSED';
      $('overlaySub').textContent = 'Press P or RESUME to continue.';
    } else if (levelClearAt && gameTimeMs < levelClearAt) {
      overlay.hidden = false;
      $('overlayTitle').textContent = 'MAZE CLEARED';
      $('overlaySub').textContent = 'Loading the next run…';
    } else {
      overlay.hidden = true;
    }
    $('pauseBtn').textContent = paused ? '▶ RESUME' : 'Ⅱ PAUSE';
    $('runStatus').textContent = gameOver ? 'GAME OVER' : paused ? 'SIMULATION PAUSED' : (levelClearAt ? 'LEVEL COMPLETE' : 'SIMULATION RUNNING');
  }

  function shortPosition(actor) {
    return { x: Math.round(actor.x), y: Math.round(actor.y) };
  }

  function directionName(actor) {
    return actor.direction ? actor.direction.toUpperCase() : 'STOP';
  }

  function snapshot() {
    const actors = [pacman, ...ghosts];
    const entityStates = actors.map((actor) => {
      const isPlayer = actor.role === 'player';
      const tile = shortPosition(actor);
      let state = actor.state;
      if (actor.returning) state = 'eyes-returning';
      else if (!isPlayer && frightenedUntilMs > gameTimeMs) state = 'frightened';
      else if (isPlayer && gameTimeMs < invulnerableUntilMs) state = 'spawn-grace';
      return {
        id: actor.id,
        label: actor.label,
        role: actor.role,
        color: actor.color,
        x: Number(actor.x.toFixed(3)),
        y: Number(actor.y.toFixed(3)),
        tile,
        pixel: { x: Math.round((actor.x + 0.5) * SOURCE_TILE), y: Math.round((actor.y + 0.5) * SOURCE_TILE) },
        direction: actor.direction || 'stop',
        state,
        speedTilesPerSecond: Number(actor.speed.toFixed(2)),
        ...(actor.target ? { movingTo: { x: actor.target.x, y: actor.target.y } } : {}),
      };
    });
    return {
      timestamp: new Date().toISOString(),
      elapsedMs: Math.round(wallClockMs),
      gameElapsedMs: Math.round(gameTimeMs),
      frame: frameCount,
      status: gameOver ? 'game-over' : paused ? 'paused' : levelClearAt ? 'level-cleared' : 'running',
      level,
      score,
      lives,
      maze: { width: WIDTH, height: HEIGHT, tileSize: SOURCE_TILE, tunnelRows: [14], coordinateSystem: 'zero-based tile coordinate; continuous x/y is measured in tile units' },
      entities: entityStates,
      pelletsRemaining,
      pelletsTotal,
      pelletRows: pelletGrid.map((row) => row.join('')),
      lastPellet: lastPelletAt,
      frightened: frightenedUntilMs > gameTimeMs,
      frightenedMsRemaining: Math.max(0, Math.round(frightenedUntilMs - gameTimeMs)),
      recentEventCount: events.length,
      events: events.slice(-20),
      newEvents: eventOutbox.slice(),
    };
  }

  async function postSnapshot() {
    if (apiSendPending || !mazeRows.length) return;
    apiSendPending = true;
    const snap = snapshot();
    try {
      const response = await fetch('/api/state', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(snap),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const sent = new Set(snap.newEvents.map((event) => event.id));
      eventOutbox = eventOutbox.filter((event) => !sent.has(event.id));
      apiSuccessTimes.push(performance.now());
      const cutoff = performance.now() - 1000;
      apiSuccessTimes = apiSuccessTimes.filter((value) => value >= cutoff);
      apiOnline = true;
      lastSnapshot = { ...snap, newEvents: undefined };
    } catch (error) {
      apiOnline = false;
      $('connectionText').textContent = 'API RETRYING';
    } finally {
      apiSendPending = false;
    }
  }

  function renderEntities() {
    const actors = [pacman, ...ghosts];
    const rows = actors.map((actor) => {
      const tile = shortPosition(actor);
      let status = actor.state;
      if (actor.returning) status = 'eyes';
      else if (actor.role === 'ghost' && frightenedUntilMs > gameTimeMs) status = 'frightened';
      return `<tr>
        <td><span class="entity-name"><i class="entity-dot" style="--entity-color:${actor.color}"></i>${actor.label}</span><span class="role-tag">${actor.role === 'player' ? 'PLAYER 01' : `GHOST / ${status.toUpperCase()}`}</span></td>
        <td><span class="tile-coord">${String(tile.x).padStart(2, '0')}, ${String(tile.y).padStart(2, '0')}</span></td>
        <td class="heading-cell">${directionName(actor)}</td>
      </tr>`;
    }).join('');
    $('entityRows').innerHTML = rows;
    const tile = shortPosition(pacman);
    $('tileExample').textContent = `x: ${tile.x} · y: ${tile.y}`;
    $('floatExample').textContent = `x: ${pacman.x.toFixed(3)} · y: ${pacman.y.toFixed(3)}`;
    $('pixelExample').textContent = `x: ${Math.round((pacman.x + 0.5) * SOURCE_TILE)} · y: ${Math.round((pacman.y + 0.5) * SOURCE_TILE)}`;
  }

  function renderEvents() {
    if (!eventDirty) return;
    $('eventCount').textContent = String(events.length).padStart(2, '0');
    if (!events.length) {
      $('eventList').innerHTML = '<div class="empty-state">Waiting for game events…</div>';
    } else {
      $('eventList').innerHTML = events.slice(-12).reverse().map((event) => `
        <div class="event-item ${event.kind || 'normal'}">
          <span class="event-time">${event.clock}</span><i class="event-dot"></i>
          <div class="event-copy">${escapeHtml(event.title)} <small>${escapeHtml(event.detail)}</small></div>
        </div>`).join('');
    }
    eventDirty = false;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  }

  function updateDashboard(force = false) {
    if (!pacman) return;
    if (!force && wallClockMs - lastUiAt < 85) return;
    lastUiAt = wallClockMs;
    $('scoreValue').textContent = String(score).padStart(6, '0');
    $('pelletValue').textContent = String(pelletsRemaining).padStart(3, '0');
    $('livesValue').textContent = String(lives).padStart(2, '0');
    $('levelValue').textContent = `LEVEL ${String(level).padStart(2, '0')}`;
    const cleared = pelletsTotal ? ((pelletsTotal - pelletsRemaining) / pelletsTotal * 100) : 0;
    $('clearedValue').textContent = `${Math.floor(cleared)}%`;
    $('pelletProgress').style.width = `${cleared}%`;
    $('clockLabel').textContent = `FRAME ${String(frameCount).padStart(6, '0')}`;
    $('apiHz').textContent = String(apiSuccessTimes.length).padStart(2, '0');
    $('connectionDot').style.background = apiOnline ? 'var(--green)' : '#f4c45a';
    $('connectionDot').style.boxShadow = apiOnline ? '0 0 10px #63e6ad66' : '0 0 10px #f4c45a66';
    $('connectionText').textContent = apiOnline ? 'API CONNECTED' : 'CONNECTING API';
    $('connectionText').parentElement.classList.toggle('connected', apiOnline);
    $('frightenedValue').textContent = frightenedUntilMs > gameTimeMs ? `${(Math.max(0, frightenedUntilMs - gameTimeMs) / 1000).toFixed(1)}s LEFT` : 'OFF';
    $('pelletCoord').textContent = lastPelletAt ? `(${lastPelletAt.x}, ${lastPelletAt.y}) · ${lastPelletAt.type.toUpperCase()}` : 'NO PELLET EATEN';
    renderEntities();
    renderEvents();
    const snap = snapshot();
    delete snap.newEvents;
    $('jsonPreview').textContent = JSON.stringify(snap, null, 2);
    syncOverlay();
  }

  function animationFrameHandler(now) {
    if (!lastFrameAt) lastFrameAt = now;
    let dt = Math.min(0.05, Math.max(0, (now - lastFrameAt) / 1000));
    lastFrameAt = now;
    wallClockMs += dt * 1000;
    frameCount += 1;
    if (!paused && !gameOver && mazeRows.length) {
      gameTimeMs += dt * 1000;
      updateGame(dt);
      recordTrails();
    }
    drawBoard();
    updateDashboard();
    if (wallClockMs - lastApiAt >= 100) {
      lastApiAt = wallClockMs;
      postSnapshot();
    }
    animationFrame = requestAnimationFrame(animationFrameHandler);
  }

  function togglePause() {
    if (gameOver || levelClearAt) return;
    paused = !paused;
    addEvent(paused ? 'pause' : 'resume', paused ? 'SIMULATION PAUSED' : 'SIMULATION RESUMED', paused ? 'Game clock frozen' : 'Game clock resumed', 'normal');
    syncOverlay();
    updateDashboard(true);
  }

  function restartGame() {
    initialState();
    addEvent('restart', 'RUN RESTARTED', 'Score, lives and pellets reset', 'success');
    syncOverlay();
  }

  function queueDirection(direction) {
    if (!pacman || paused || gameOver) return;
    pacman.wantedDirection = direction;
    if (!pacman.target && canMove(pacman.tileX, pacman.tileY, direction, 'player')) {
      pacman.direction = direction;
    }
  }

  document.addEventListener('keydown', (event) => {
    const key = event.key.toLowerCase();
    const keyMap = { arrowup: 'up', w: 'up', arrowdown: 'down', s: 'down', arrowleft: 'left', a: 'left', arrowright: 'right', d: 'right' };
    if (keyMap[key]) {
      event.preventDefault();
      queueDirection(keyMap[key]);
      return;
    }
    if (key === 'p' || key === ' ') {
      event.preventDefault();
      togglePause();
    } else if (key === 'r') {
      event.preventDefault();
      restartGame();
    } else if (key === 'g') {
      showGrid = !showGrid;
      $('gridToggle').setAttribute('aria-pressed', String(showGrid));
    } else if (key === 't') {
      showTrails = !showTrails;
      $('trailToggle').setAttribute('aria-pressed', String(showTrails));
    }
  });

  $('pauseBtn').addEventListener('click', togglePause);
  $('restartBtn').addEventListener('click', restartGame);
  $('gridToggle').addEventListener('click', () => {
    showGrid = !showGrid;
    $('gridToggle').setAttribute('aria-pressed', String(showGrid));
    drawBoard();
  });
  $('trailToggle').addEventListener('click', () => {
    showTrails = !showTrails;
    $('trailToggle').setAttribute('aria-pressed', String(showTrails));
    drawBoard();
  });
  $('viewJsonBtn').addEventListener('click', () => {
    $('jsonPanel').scrollIntoView({ behavior: 'smooth', block: 'center' });
    $('jsonPreview').classList.add('highlight-json');
    setTimeout(() => $('jsonPreview').classList.remove('highlight-json'), 800);
  });
  $('copyJsonBtn').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(snapshot(), null, 2));
      showToast('Current state copied to clipboard');
    } catch {
      showToast('Clipboard unavailable — select the JSON preview manually');
    }
  });
  document.querySelectorAll('[data-endpoint]').forEach((button) => {
    button.addEventListener('click', () => {
      const path = button.dataset.endpoint;
      if (path === '/api/stream') window.open(path, '_blank', 'noopener');
      else window.open(path, '_blank', 'noopener');
    });
  });

  let toastTimer;
  function showToast(message) {
    let toast = document.querySelector('.toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'toast';
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 1900);
  }

  async function initialize() {
    try {
      const response = await fetch('/api/maze', { cache: 'no-store' });
      if (!response.ok) throw new Error(`Maze API returned ${response.status}`);
      const maze = await response.json();
      if (maze.width !== WIDTH || maze.height !== HEIGHT || !Array.isArray(maze.rows) || !Array.isArray(maze.playerWalkableRows) || !Array.isArray(maze.ghostWalkableRows)) throw new Error('Maze dimensions do not match the game canvas');
      mazeRows = maze.rows;
      playerWalkableRows = maze.playerWalkableRows;
      ghostWalkableRows = maze.ghostWalkableRows;
      if (mazeRows.some((row) => row.length !== WIDTH) || playerWalkableRows.some((row) => row.length !== WIDTH) || ghostWalkableRows.some((row) => row.length !== WIDTH)) throw new Error('Maze row length is invalid');
      initialState();
      apiOnline = true;
      const stream = new EventSource('/api/stream');
      stream.addEventListener('state', () => {
        apiOnline = true;
      });
      stream.onopen = () => { apiOnline = true; };
      stream.onerror = () => { apiOnline = false; };
    } catch (error) {
      $('connectionText').textContent = 'API OFFLINE';
      $('runStatus').textContent = 'API UNAVAILABLE';
      console.error(error);
      return;
    }
    setCanvasScale();
    window.addEventListener('resize', setCanvasScale);
    animationFrame = requestAnimationFrame(animationFrameHandler);
  }

  initialize();
})();
