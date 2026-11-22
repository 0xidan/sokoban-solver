// A push-optimal Sokoban solver for make.fun's generated levels.
//
// A generated level is a hallway with rooms off it, and a box pushed into the hallway can never come back, so every
// box stays in its room and the level's push optimum is the sum of its rooms' optima. This solver cuts the level into
// its rooms the generator's way (the dead squares of the player's row are cut out of the floor, and each part left
// that holds boxes is a room), then solves each room on its own with an A* over pushes: a state is the boxes'
// squares and the player's reachable region (named by its lowest square), each push costs 1, and the estimate is
// each box's shortest push distance to a goal, summed — precomputed per room by a reverse search over pulls that
// ignores the other boxes, so it never overstates and the first solved state the search settles is push-optimal.
// Pushes that land a box on a dead square (one no goal is reachable from, the same reverse search) or freeze it in
// a 2 x 2 block of walls and boxes are pruned; neither can sit on a solution. While a room is solved the other
// rooms' boxes stand as walls (on their goals once their room is done) and the player walks on from where the last
// room left it, so the rooms' moves join into the level's. A level that is no set of rooms falls back to the same
// search over the whole board. Everything is budgeted by states searched, never the clock, and every tie breaks by
// square and creation order, so two runs answer byte for byte alike.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The four directions: [dx, dy, the walk's letter, the push's letter]. */
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** The most states one room's search creates before it gives up, and one whole-board search's: work, never the clock. */
const ROOM_STATES = 500_000;
const BOARD_STATES = 600_000;

/**
 * A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player, + player on a goal), each line
 * a row: its width and height, walls (1 a wall, short rows and squares outside padded as walls, the outermost frame
 * always a wall), goals, boxes (sorted) and player.
 */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Each square's shortest push distance to a goal: the fewest pushes that bring a lone box from there onto some goal,
 * by a reverse search from the goals over pulls (a box at y is pushed by dir onto x when the square behind y is open
 * for the player). -1 where no push sequence reaches a goal: a dead square, off every solution.
 */
function pushDistances(w, wall, goals) {
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [];
  for (const g of goals) {
    dist[g] = 0;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
      const y = x - d;
      if (dist[y] === -1 && !wall[y] && !wall[y - d]) {
        dist[y] = dist[x] + 1;
        queue.push(y);
      }
    }
  }
  return dist;
}

/** Whether the box just pushed to `t` sits in a frozen 2 x 2 block of walls and boxes with a box off its goal. */
function frozenBlock(w, wall, occupied, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (the first direction first on ties). */
function walkTo(w, wall, occupied, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, the case aside) from `from` on a board `w` wide. */
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The push-optimal solution of one board (a room, or a whole level as the fallback) as a moves string, searching at
 * most `maxStates` states: null when the search gives up or a box starts on a dead square. A* over pushes with the
 * summed push-distance estimate, which is consistent (one push moves one box one square, changing its distance by
 * at most the push's cost), so a state popped from the heap already holds its fewest pushes.
 */
function solvePush(w, wall, goals, boxes0, player0, maxStates) {
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const dist = pushDistances(w, wall, goals);
  let h0 = 0;
  for (const b of boxes0) {
    if (dist[b] < 0) return null;
    h0 += dist[b];
  }
  if (h0 === 0) return '';
  const occupied = new Uint8Array(size);
  const stack = new Int32Array(size);
  // Two flood fills, each with its own marks: one names the region a state is expanded from (its squares are where
  // the player can stand to push), the other the region a push leads to (its lowest square names the new state).
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  let seenStamp = 0;
  let markStamp = 0;
  const flood = (marks, stamp, from) => {
    let low = from;
    let n = 0;
    stack[n++] = from;
    marks[from] = stamp;
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
        if (wall[q] === 0 && occupied[q] === 0 && marks[q] !== stamp) {
          marks[q] = stamp;
          stack[n++] = q;
        }
      }
    }
    return low;
  };
  const keyOf = (boxes, low) => String.fromCharCode(low, ...boxes);
  // The states in the order created: their boxes, the square the player stands on, pushes made, the estimate, the
  // parent state and the state's key. The heap orders by pushes + estimate, then more pushes first, then older first.
  const sb = [boxes0];
  const sp = [player0];
  const sg = [0];
  const sh = [h0];
  const sparent = [-1];
  const skey = [keyOf(boxes0, flood(mark, ++markStamp, player0))];
  const best = new Map([[skey[0], 0]]);
  const heap = [[h0, 0, 0]];
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const pushHeap = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const popHeap = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  let found = -1;
  while (heap.length > 0) {
    const [, g, s] = popHeap();
    if (best.get(skey[s]) !== g) continue;
    if (sh[s] === 0) {
      found = s;
      break;
    }
    if (sb.length >= maxStates) return null;
    const boxes = sb[s];
    for (const b of boxes) occupied[b] = 1;
    const here = ++seenStamp;
    flood(seen, here, sp[s]);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = k === 0 ? -w : k === 1 ? w : k === 2 ? -1 : 1;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || dist[t] < 0) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let nb = null;
        let key = '';
        if (!frozen) {
          nb = boxes.slice();
          nb[bi] = t;
          nb.sort((x, y) => x - y);
          key = keyOf(nb, flood(mark, ++markStamp, b));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const ng = g + 1;
        const seenG = best.get(key);
        if (seenG !== undefined && seenG <= ng) continue;
        let nh = 0;
        for (const x of nb) nh += dist[x];
        const t2 = sb.length;
        sb.push(nb);
        sp.push(b);
        sg.push(ng);
        sh.push(nh);
        sparent.push(s);
        skey.push(key);
        best.set(key, ng);
        pushHeap([ng + nh, ng, t2]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // The path of states, then the moves: walk to each push's square, then push.
  const path = [];
  for (let s = found; s > 0; s = sparent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = player0;
  let boxes = boxes0;
  for (const s of path) {
    const nextBoxes = sb[s];
    const from = boxes.find((b) => !nextBoxes.includes(b));
    const to = nextBoxes.find((b) => !boxes.includes(b));
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return moves;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no goal) out of the
 * floor, and each part left that holds a box is a room, its squares, boxes and goals. Null when some part holds more
 * or fewer boxes than goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const dist = pushDistances(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && dist[i] < 0) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** The moves for a level in XSB: a string of u, d, l and r (an upper letter marks a push; case is not checked). */
function solve(text) {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      const sol = solvePush(level.w, wall, room.goals.slice().sort((a, b) => a - b), room.boxes.slice().sort((a, b) => a - b), player, ROOM_STATES);
      if (sol === null) {
        ok = false;
        break;
      }
      moves += sol;
      player = endOf(level.w, player, sol);
      room.done = true;
    }
    if (ok) return moves;
  }
  return solvePush(level.w, level.wall, level.goals, level.boxes, level.player, BOARD_STATES) ?? '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
