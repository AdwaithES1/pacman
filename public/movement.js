'use strict';

function canEnterTile(x, y, type, width, height, playerWalkableRows, ghostWalkableRows) {
  if (y < 0 || y >= height || x < 0 || x >= width) return false;
  const mask = type === 'ghost' ? ghostWalkableRows : playerWalkableRows;
  const value = mask[y]?.[x] || '0';
  return value === '1' || (type === 'ghost' && value === '2');
}

function actorStepTarget(actor, direction, destination, canEnter) {
  const next = destination(actor.tileX, actor.tileY, direction);
  if (!next || !canEnter(next.x, next.y, actor.role === 'ghost' ? 'ghost' : 'player')) return null;
  const warp = next.y === 14 && Math.abs(next.x - actor.tileX) > 1;
  return { ...next, warp };
}

const PacmanMovement = { canEnterTile, actorStepTarget };
if (typeof module === 'object' && module.exports) module.exports = PacmanMovement;
else globalThis.PacmanMovement = PacmanMovement;
