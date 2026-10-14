'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { actorStepTarget, canEnterTile } = require('../public/movement.js');

const WIDTH = 28;
const HEIGHT = 31;
const playerWalkableRows = Array(HEIGHT).fill('0'.repeat(WIDTH));
const ghostWalkableRows = playerWalkableRows.slice();
ghostWalkableRows[14] = `${'0'.repeat(14)}2${'0'.repeat(13)}`;

function canEnter(x, y, type) {
  return canEnterTile(x, y, type, WIDTH, HEIGHT, playerWalkableRows, ghostWalkableRows);
}

function destination(x, y, direction) {
  const delta = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[direction];
  return delta ? { x: x + delta[0], y: y + delta[1] } : null;
}

test('normal-game ghost movement rejects a wall target and permits ghost-only floor', () => {
  const ghost = { role: 'ghost', tileX: 14, tileY: 15 };
  assert.equal(actorStepTarget(ghost, 'down', destination, canEnter), null);
  assert.deepEqual(actorStepTarget(ghost, 'up', destination, canEnter), { x: 14, y: 14, warp: false });
  assert.equal(actorStepTarget({ role: 'player', tileX: 14, tileY: 15 }, 'up', destination, canEnter), null);
});
