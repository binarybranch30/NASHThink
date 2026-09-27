// Browser end-to-end test for sarthink_graph.html against the real graph CSVs.
//
// Needs a running server and Playwright (not a project dependency; install it anywhere):
//   HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py --port 8765 &
//   npm install --prefix /tmp/pw playwright && npx --prefix /tmp/pw playwright install chromium
//   NODE_PATH=/tmp/pw/node_modules node scripts/tests/graph_ui_e2e.mjs [http://127.0.0.1:8765/]
//
// Set SARTHINK_E2E_SEMANTIC=1 to also run a real memory search (loads the embedding model on CPU).
// Error/fallback states are exercised by rewriting the CSV responses in the browser; nothing on
// disk is modified. Memory Home, Ask/search and Insights answers are stubbed with synthetic text, and
// no check prints archive-derived text (only counts, ids and booleans).
//
// The page opens on Memory Home; the graph flows below enter the workspace first (ready()).
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const BASE = process.argv[2] || process.env.SARTHINK_URL || 'http://127.0.0.1:8765/';
const SHOTS = process.env.SARTHINK_E2E_SHOTS || '';   // directory for screenshots, optional
const REAL_SEMANTIC = process.env.SARTHINK_E2E_SEMANTIC === '1';

let failures = 0;
const results = [];
function check(name, ok, detail = '') {
  results.push(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) failures++;
}

async function shot(page, name) {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png` });
}

async function open(browser, { url = BASE, viewport = { width: 1440, height: 900 }, route } = {}) {
  const page = await browser.newPage({ viewport });
  page.errors = [];
  page.on('pageerror', e => page.errors.push(e.message));
  if (route) {
    await route(page);
    // Stubbed routes still in flight must not outlive the page.
    const close = page.close.bind(page);
    page.close = async opts => { await page.unrouteAll({ behavior: 'ignoreErrors' }).catch(() => {}); return close(opts); };
  }
  await page.goto(url);
  return page;
}

// Waits for the graph; by default then opens the graph workspace (the page starts on Memory Home).
async function ready(page, { view = 'graph' } = {}) {
  await page.waitForFunction(() => window.__sarthink && window.__sarthink.pGeo && document.body.dataset.graph === 'ready', null, { timeout: 30000 });
  if (view === 'graph' && await page.evaluate(() => document.body.dataset.view !== 'graph')) await page.click('#btn-explore');
  await settle(page);
}

// Waits for camera flights and control damping to finish.
async function settle(page) {
  await page.waitForFunction(() => !window.__sarthink.tween, null, { timeout: 5000 });
  await page.waitForTimeout(700);
}

// Screen position of node i (CSS pixels) and whether it is in front of the camera.
function screenOf(page, i) {
  return page.evaluate(i => {
    const S = window.__sarthink, n = S.nodes[i], cam = S.camera;
    cam.updateMatrixWorld();
    const p = cam.projectionMatrix.elements, v = cam.matrixWorldInverse.elements;
    const mv = [0, 1, 2, 3].map(r => v[r] * n.x + v[4 + r] * n.y + v[8 + r] * n.z + v[12 + r]);
    const clip = [0, 1, 2, 3].map(r => p[r] * mv[0] + p[4 + r] * mv[1] + p[8 + r] * mv[2] + p[12 + r] * mv[3]);
    return { x: (clip[0] / clip[3] + 1) / 2 * innerWidth, y: (1 - clip[1] / clip[3]) / 2 * innerHeight, front: clip[3] > 0 };
  }, i);
}

// Points the camera at a mid-degree thread from close up (as a user would after zooming in), then
// returns a well-connected node that is on screen, clear of the panels and not overlapped by other
// nodes, so a click on it is unambiguous.
async function isolatedNode(page, minDegree = 2) {
  for (const frac of [0.25, 0.12, 0.06, 0.03]) {
    await page.evaluate(frac => {
      const S = window.__sarthink;
      const cands = S.nodes.filter(n => n.kind === 'thread' && n.links.length >= 3 && n.links.length <= 12 && S.pAlpha[n.i] > 0);
      const n = cands[Math.floor(cands.length / 2)] || S.nodes[0];
      S.tween = null;
      S.controls.target.set(n.x, n.y, n.z);
      const d = S.camera.position.clone().sub(S.home.target).normalize().multiplyScalar(S.bounds.radius * frac);
      S.camera.position.set(n.x + d.x, n.y + d.y, n.z + d.z);
      S.controls.update();
    }, frac);
    await page.waitForTimeout(400);
    const found = await findIsolated(page, minDegree);
    if (found) return found;
  }
  return null;
}

function findIsolated(page, minDegree) {
  return page.evaluate(minDegree => {
    const S = window.__sarthink, cam = S.camera;
    cam.updateMatrixWorld();
    const p = cam.projectionMatrix.elements, v = cam.matrixWorldInverse.elements;
    const scr = S.nodes.map(n => {
      const mv = [0, 1, 2, 3].map(r => v[r] * n.x + v[4 + r] * n.y + v[8 + r] * n.z + v[12 + r]);
      const c = [0, 1, 2, 3].map(r => p[r] * mv[0] + p[4 + r] * mv[1] + p[8 + r] * mv[2] + p[12 + r] * mv[3]);
      return [(c[0] / c[3] + 1) / 2 * innerWidth, (1 - c[1] / c[3]) / 2 * innerHeight, c[3]];
    });
    const blocked = el => { const r = document.getElementById(el).getBoundingClientRect(); return r; };
    const side = blocked('sidebar'), right = document.getElementById('rightcol').getBoundingClientRect();
    const order = S.nodes.map((n, i) => i).filter(i => S.pAlpha[i] > 0 && S.nodes[i].links.length >= minDegree)
      .sort((a, b) => S.nodes[b].links.length - S.nodes[a].links.length);
    for (const i of order) {
      const [x, y, w] = scr[i];
      if (w <= 0 || x < side.right + 30 || x > innerWidth - 40 || y < 80 || y > innerHeight - 60) continue;
      if (right.width && x > right.left - 30) continue;
      let clear = true;
      for (let j = 0; j < scr.length && clear; j++) {
        if (j !== i && S.pAlpha[j] > 0 && scr[j][2] > 0 && Math.hypot(scr[j][0] - x, scr[j][1] - y) < 14) clear = false;
      }
      if (clear) return { i, x, y, degree: S.nodes[i].links.length };
    }
    return null;
  }, minDegree);
}

const state = page => page.evaluate(() => {
  const S = window.__sarthink;
  return {
    selected: S.selected, hover: S.hover, visibleNodes: S.visibleNodes, visibleEdges: S.visibleEdges,
    nodes: S.nodes.length, edges: S.edges.length, platforms: [...S.filter.platforms], kinds: [...S.filter.kinds],
    semSet: S.sem.set ? S.sem.set.size : 0, findQ: S.find.q, detailsHidden: document.getElementById('details').hidden,
    semHidden: document.getElementById('sem-panel').hidden,
    focusEdges: S.focusEdges ? S.focusEdges.geometry.attributes.position.count / 2 : 0,
    focusPoints: S.focusPoints ? S.focusPoints.geometry.attributes.position.count : 0,
    cam: S.camera.position.toArray(), target: S.controls.target.toArray(),
    home: S.home.position.toArray(), homeTarget: S.home.target.toArray(), layoutNote: S.layoutNote,
    timeActive: !!(S.time && (S.time.i0 > 0 || S.time.i1 < S.time.months.length - 1)),
    hash: location.hash,
  };
});

// After a camera flight, the selection and all its neighbours must be on screen and not under a panel.
function selectionInFreeArea(page) {
  return page.evaluate(() => {
    const S = window.__sarthink, cam = S.camera, i = S.selected;
    cam.updateMatrixWorld();
    const p = cam.projectionMatrix.elements, v = cam.matrixWorldInverse.elements;
    const side = document.getElementById('sidebar').getBoundingClientRect();
    const rc = document.getElementById('rightcol').getBoundingClientRect();
    const idx = [i, ...S.nodes[i].links.map(k => { const e = S.edges[k]; return e.s === i ? e.t : e.s; })];
    const bad = idx.filter(j => {
      const n = S.nodes[j];
      const mv = [0, 1, 2, 3].map(r => v[r] * n.x + v[4 + r] * n.y + v[8 + r] * n.z + v[12 + r]);
      const c = [0, 1, 2, 3].map(r => p[r] * mv[0] + p[4 + r] * mv[1] + p[8 + r] * mv[2] + p[12 + r] * mv[3]);
      const x = (c[0] / c[3] + 1) / 2 * innerWidth, y = (1 - c[1] / c[3]) / 2 * innerHeight;
      return c[3] <= 0 || x < side.right || x > rc.left || y < 0 || y > innerHeight;
    });
    return { ok: bad.length === 0, n: idx.length, bad: bad.length };
  });
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

async function mainFlow(browser) {
  const page = await open(browser);
  await ready(page);
  await shot(page, '01_initial');
  let st = await state(page);
  check('graph loads with nodes and edges', st.nodes > 0 && st.edges > 0, `${st.nodes} nodes, ${st.edges} edges`);
  check('no layout fallback for the real export', st.layoutNote === '', st.layoutNote);

  // 1. Exported positions are used verbatim (GPU buffer == CSV values).
  const csvCheck = await page.evaluate(async () => {
    const S = window.__sarthink;
    const txt = await (await fetch('./processed_data/graph/cosmograph_nodes.csv', { cache: 'no-store' })).text();
    const rows = Papa.parse(txt, { header: true, skipEmptyLines: true }).data;
    const pos = S.pGeo.attributes.position.array;
    let bad = 0, spanY = [Infinity, -Infinity], spanX = [Infinity, -Infinity];
    rows.forEach(r => {
      const n = S.byId.get(r.id);
      const i = n.i;
      if (Math.abs(pos[i * 3] - +r.layout_x) > 1e-3 || Math.abs(pos[i * 3 + 1] - +r.layout_y) > 1e-3 || Math.abs(pos[i * 3 + 2] - +r.layout_z) > 1e-3) bad++;
      spanX = [Math.min(spanX[0], +r.layout_x), Math.max(spanX[1], +r.layout_x)];
      spanY = [Math.min(spanY[0], +r.layout_y), Math.max(spanY[1], +r.layout_y)];
    });
    return { bad, rows: rows.length, spanX: spanX[1] - spanX[0], spanY: spanY[1] - spanY[0] };
  });
  check('GPU positions equal CSV layout_x/y/z for every node', csvCheck.bad === 0, `${csvCheck.bad}/${csvCheck.rows} differ`);
  check('layout is not a line (x and y spans comparable)', Math.min(csvCheck.spanX, csvCheck.spanY) / Math.max(csvCheck.spanX, csvCheck.spanY) > 0.2,
    `x span ${csvCheck.spanX.toFixed(0)}, y span ${csvCheck.spanY.toFixed(0)}`);

  // Every node is inside the initial viewport.
  const framed = await page.evaluate(() => {
    const S = window.__sarthink, cam = S.camera;
    cam.updateMatrixWorld();
    const p = cam.projectionMatrix.elements, v = cam.matrixWorldInverse.elements;
    let out = 0;
    for (const n of S.nodes) {
      const mv = [0, 1, 2, 3].map(r => v[r] * n.x + v[4 + r] * n.y + v[8 + r] * n.z + v[12 + r]);
      const c = [0, 1, 2, 3].map(r => p[r] * mv[0] + p[4 + r] * mv[1] + p[8 + r] * mv[2] + p[12 + r] * mv[3]);
      if (c[3] <= 0 || Math.abs(c[0] / c[3]) > 1 || Math.abs(c[1] / c[3]) > 1) out++;
    }
    return out;
  });
  check('initial camera frames the whole graph', framed === 0, `${framed} nodes off-screen`);

  // 4. Overlays only capture pointer events inside their own box.
  const hitTest = await page.evaluate(() => {
    const at = (x, y) => document.elementFromPoint(x, y);
    const canvas = document.querySelector('#stage canvas');
    const side = document.getElementById('sidebar').getBoundingClientRect();
    const pts = {
      centre: [innerWidth / 2, innerHeight / 2],
      topStripBetweenStatsAndCommandBar: [(document.getElementById('stats').getBoundingClientRect().right + document.getElementById('omni').getBoundingClientRect().left) / 2, 28],
      rightEdgeNoPanels: [innerWidth - 30, innerHeight / 2],
      belowSidebar: [side.left + 20, Math.min(innerHeight - 20, side.bottom + 30)],
      besideHint: [innerWidth * 0.2, innerHeight - 20],
    };
    const out = {};
    for (const [k, [x, y]] of Object.entries(pts)) out[k] = at(x, y) === canvas;
    out.insideSidebar = document.getElementById('sidebar').contains(at(side.left + 30, side.top + 20));
    const cmd = document.getElementById('omni').getBoundingClientRect();
    out.commandBarInTopbar = document.getElementById('omni').contains(at(cmd.left + cmd.width / 2, cmd.top + cmd.height / 2));
    out.homeCardGone = !document.getElementById('home').contains(at(innerWidth / 2, innerHeight * 0.3));
    return out;
  });
  for (const [k, v] of Object.entries(hitTest)) check(`pointer hit-test: ${k}`, v);

  // 3. Orbit: left-drag starting ON a node rotates the camera; it must not select or move nodes.
  const target = await isolatedNode(page);
  check('found an unambiguous node to interact with', !!target);
  const before = await state(page);
  const pos0 = await page.evaluate(() => Array.from(window.__sarthink.pGeo.attributes.position.array.slice(0, 300)));
  await page.mouse.move(target.x, target.y);
  await page.mouse.down();
  for (let k = 1; k <= 10; k++) await page.mouse.move(target.x + k * 18, target.y + k * 6);
  await page.mouse.up();
  await settle(page);
  let after = await state(page);
  const pos1 = await page.evaluate(() => Array.from(window.__sarthink.pGeo.attributes.position.array.slice(0, 300)));
  check('left-drag on a node orbits the camera', dist(before.cam, after.cam) > 1, `moved ${dist(before.cam, after.cam).toFixed(1)}`);
  check('orbit keeps the orbit target', dist(before.target, after.target) < 1e-6);
  check('orbit does not drag nodes', pos0.every((v, k) => v === pos1[k]));
  check('drag does not select', after.selected === -1);

  // Pan (right-drag) moves the target; shift-drag pans too.
  let b2 = await state(page);
  await page.mouse.move(900, 500);
  await page.mouse.down({ button: 'right' });
  for (let k = 1; k <= 8; k++) await page.mouse.move(900 + k * 15, 500 + k * 8);
  await page.mouse.up({ button: 'right' });
  await settle(page);
  let a2 = await state(page);
  check('right-drag pans', dist(b2.target, a2.target) > 1);
  b2 = a2;
  await page.keyboard.down('Shift');
  await page.mouse.move(900, 500);
  await page.mouse.down();
  for (let k = 1; k <= 8; k++) await page.mouse.move(900 - k * 15, 500);
  await page.mouse.up();
  await page.keyboard.up('Shift');
  await settle(page);
  a2 = await state(page);
  check('shift-drag pans', dist(b2.target, a2.target) > 1);

  // Zoom
  b2 = a2;
  await page.mouse.move(800, 450);
  for (let k = 0; k < 5; k++) { await page.mouse.wheel(0, -200); await page.waitForTimeout(40); }
  await settle(page);
  a2 = await state(page);
  check('wheel zooms in', dist(a2.cam, a2.target) < dist(b2.cam, b2.target) * 0.95,
    `${dist(b2.cam, b2.target).toFixed(0)} → ${dist(a2.cam, a2.target).toFixed(0)}`);

  // Escape returns to the home view.
  await page.keyboard.press('Escape');
  await settle(page);
  st = await state(page);
  check('Esc flies the camera home', dist(st.cam, st.home) < 1 && dist(st.target, st.homeTarget) < 1);

  // Hover tooltip
  const node = await isolatedNode(page);
  await page.mouse.move(node.x - 40, node.y - 40);
  await page.mouse.move(node.x, node.y, { steps: 4 });
  await page.waitForTimeout(150);
  st = await state(page);
  const tip = await page.evaluate(() => ({ show: document.getElementById('tooltip').classList.contains('show'), text: document.getElementById('tooltip').textContent }));
  check('hover picks the node under the cursor', st.hover === node.i, `hover=${st.hover} want=${node.i}`);
  check('hover shows a tooltip with platform and connections', tip.show && /connection/.test(tip.text), `${tip.text.length} chars`);
  const cursor = await page.evaluate(() => getComputedStyle(document.querySelector('#stage canvas')).cursor);
  check('pointer cursor over a node', cursor === 'pointer', cursor);

  // 5. Click selects: node + all neighbours + all its edges visible and highlighted.
  await page.mouse.click(node.x, node.y);
  await page.waitForTimeout(150);
  st = await state(page);
  check('click selects the node under the cursor', st.selected === node.i, `selected=${st.selected} want=${node.i}`);
  check('selection draws one highlighted edge per connection', st.focusEdges === node.degree, `${st.focusEdges} edges for degree ${node.degree}`);
  check('selection keeps the node and every neighbour visible', st.focusPoints === node.degree + 2, `${st.focusPoints} focus points`);  // + halo
  check('details panel opens', !st.detailsHidden);
  const dims = await page.evaluate(() => {
    const S = window.__sarthink;
    const nb = new Set([S.selected, ...S.nodes[S.selected].links.map(k => { const e = S.edges[k]; return e.s === S.selected ? e.t : e.s; })]);
    let dimmed = 0, hidden = 0, others = 0;
    S.nodes.forEach((n, i) => { if (nb.has(i)) return; others++; if (S.pAlpha[i] > 0 && S.pAlpha[i] < 0.5) dimmed++; if (S.pAlpha[i] === 0) hidden++; });
    return { dimmed, hidden, others };
  });
  check('unrelated nodes are dimmed, not hidden', dims.dimmed === dims.others && dims.hidden === 0, JSON.stringify(dims));
  const details = await page.evaluate(() => document.getElementById('details').innerText);
  check('details show connections, messages and dates', /CONNECTIONS/i.test(details) && /MESSAGES/i.test(details) && /FIRST ACTIVE/i.test(details));
  st = await state(page);
  check('selection is written to the URL hash', st.hash.startsWith('#node='), st.hash);
  await shot(page, '02_selected');

  // Clicking the same node again must keep the selection and its links (old bug: links vanished).
  const again = await screenOf(page, node.i);
  await page.mouse.click(again.x, again.y);
  await page.waitForTimeout(150);
  st = await state(page);
  check('re-clicking the selected node keeps it selected', st.selected === node.i);
  check('re-clicking keeps all its links', st.focusEdges === node.degree);

  // Orbit while selected keeps selection + links
  await page.mouse.move(again.x + 200, again.y + 120);
  await page.mouse.down();
  await page.mouse.move(again.x + 320, again.y + 150, { steps: 6 });
  await page.mouse.up();
  await settle(page);
  st = await state(page);
  check('orbiting keeps the selection and its links', st.selected === node.i && st.focusEdges === node.degree);

  // Clicking a linked item in the details panel moves the selection there.
  const linkBtn = page.locator('#d-links .nitem').first();
  const linkJ = +(await linkBtn.getAttribute('data-j'));
  await linkBtn.click();
  await settle(page);
  st = await state(page);
  check('details link selects the neighbour', st.selected === linkJ);

  // Conversation context for thread nodes (needs the index; reports either way).
  const threadIdx = await page.evaluate(() => window.__sarthink.nodes.findIndex(n => n.kind === 'thread' && n.links.length >= 2));
  await page.evaluate(i => { location.hash = '#node=' + window.__sarthink.nodes[i].id; }, threadIdx);
  await page.waitForFunction(i => window.__sarthink.selected === i, threadIdx);
  await page.waitForFunction(() => { const c = document.getElementById('d-ctx'); return c && !/Loading/.test(c.innerText); }, null, { timeout: 15000 });
  const ctx = await page.evaluate(() => document.getElementById('d-ctx').innerText);
  check('thread details show conversation context or an honest reason', /indexed chunk|no chunks|needs the local API|isn’t available/.test(ctx));

  // Click on empty space deselects.
  await settle(page);
  await page.mouse.click(Math.round(await page.evaluate(() => innerWidth)) - 400, 870);
  await page.waitForTimeout(150);
  st = await state(page);
  check('click on empty space deselects', st.selected === -1 && st.detailsHidden, `selected=${st.selected}`);

  // Platform filter
  await page.keyboard.press('Escape');
  await settle(page);
  const plats = await page.$$eval('#plat-list .frow', els => els.map(e => e.dataset.p));
  check('legend lists every platform in the data', plats.length === (await state(page)).platforms.length, plats.join(','));
  const counts = await page.evaluate(() => { const c = {}; window.__sarthink.nodes.forEach(n => c[n.platform] = (c[n.platform] || 0) + 1); return c; });
  const p0 = plats[0];
  await page.click(`#plat-list .frow[data-p="${p0}"]`);
  st = await state(page);
  check(`toggling ${p0} off hides exactly its nodes`, st.visibleNodes === st.nodes - counts[p0], `${st.visibleNodes} visible`);
  const pressed = await page.getAttribute(`#plat-list .frow[data-p="${p0}"]`, 'aria-pressed');
  check('platform toggle exposes aria-pressed', pressed === 'false');
  const hiddenPick = await page.evaluate(p => {
    const S = window.__sarthink; return S.nodes.every((n, i) => n.platform !== p || S.pAlpha[i] === 0);
  }, p0);
  check('filtered nodes are not drawn', hiddenPick);
  const statsTxt = await page.evaluate(() => document.getElementById('stats').innerText);
  check('stats show filtered counts', /showing/.test(statsTxt), statsTxt.replace(/\s+/g, ' '));
  await shot(page, '03_filtered');
  // only
  await page.hover(`#plat-list .fwrap:last-child .frow`);
  await page.click(`#plat-list .fwrap:last-child .only`);
  st = await state(page);
  check('"only" shows a single platform', st.platforms.length === 1 && st.platforms[0] === plats[plats.length - 1]);
  // Hide all platforms → empty state
  await page.click(`#plat-list .frow[data-p="${plats[plats.length - 1]}"]`);
  const empty = await page.evaluate(() => !document.getElementById('empty-state').hidden);
  check('empty state when filters hide everything', empty);
  await page.click('#empty-reset');
  st = await state(page);
  check('empty-state button restores filters', st.visibleNodes === st.nodes);

  // Node type filter
  await page.click('#kind-list .frow[data-k="thread"]');
  st = await state(page);
  const users = await page.evaluate(() => window.__sarthink.nodes.filter(n => n.kind === 'user').length);
  check('node type filter hides threads', st.visibleNodes === users && st.visibleEdges === 0);
  await page.click('#kind-list .frow[data-k="thread"]');

  // Timeline
  const hasTime = await page.evaluate(() => !!window.__sarthink.time);
  check('timeline enabled when the export has dates', hasTime);
  if (hasTime) {
    await page.click('#tl-presets button[data-m="3"]');
    st = await state(page);
    check('timeline preset narrows the graph', st.timeActive && st.visibleNodes < st.nodes && st.visibleNodes > 0, `${st.visibleNodes} visible`);
    const tlOK = await page.evaluate(() => {
      const S = window.__sarthink, t = S.time;
      const t0 = t.months[t.i0];
      return S.nodes.every((n, i) => !S.nodeVis[i] || n.kind !== 'thread' || n.last >= t0);
    });
    check('timeline hides threads that ended before the range', tlOK);
    await shot(page, '04_timeline');
  }

  // Esc restores everything
  await page.keyboard.press('Escape');
  await settle(page);
  st = await state(page);
  check('Esc restores all filters', st.visibleNodes === st.nodes && st.visibleEdges === st.edges && !st.timeActive);

  // Find
  await page.keyboard.press('/');
  const focused = await page.evaluate(() => document.activeElement.id);
  check('"/" focuses the find box', focused === 'find-q');
  const q = await page.evaluate(() => { const n = window.__sarthink.nodes.find(n => n.kind === 'user' && n.links.length > 2); return n.label.slice(0, 5); });
  await page.keyboard.type(q);
  st = await state(page);
  const findItems = await page.$$eval('#find-list .fitem', els => els.length);
  check('find lists and highlights matches', findItems > 0 && st.findQ === q.toLowerCase());
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await settle(page);
  st = await state(page);
  check('find: arrow + Enter selects and flies', st.selected >= 0 && !st.detailsHidden);
  let framed2 = await selectionInFreeArea(page);
  check('fly-to frames the node and all neighbours clear of the panels', framed2.ok, `${framed2.bad}/${framed2.n} outside`);
  await page.keyboard.press('Escape');
  await settle(page);
  st = await state(page);
  const findVal = await page.inputValue('#find-q');
  check('Esc from the find box clears it and resets', st.selected === -1 && findVal === '' && st.findQ === '');

  // Help overlay
  await page.keyboard.press('?');
  let helpVisible = await page.isVisible('#help');
  check('"?" opens the help overlay', helpVisible);
  await page.keyboard.press('Escape');
  helpVisible = await page.isVisible('#help');
  check('Esc closes help first', !helpVisible);
  await page.click('#btn-help');
  check('Help button opens help', await page.isVisible('#help'));
  await page.click('#help-close');

  // Reset button restores selection + filters + camera
  const n2 = await isolatedNode(page);
  await page.mouse.click(n2.x, n2.y);
  await page.click(`#plat-list .frow[data-p="${plats[plats.length - 1]}"]`);
  await page.click('#btn-reset');
  await settle(page);
  st = await state(page);
  check('Reset button restores the full view', st.selected === -1 && st.visibleNodes === st.nodes && dist(st.cam, st.home) < 1 && st.hash === '');

  // Refresh keeps the selected node (URL hash) and reloads the CSVs uncached.
  const n3 = await isolatedNode(page);   // the camera moved home, so look again
  await page.mouse.click(n3.x, n3.y);
  st = await state(page);
  check('click after reset selects again', st.selected === n3.i);
  const selId = await page.evaluate(() => window.__sarthink.nodes[window.__sarthink.selected].id);
  await page.reload();
  await ready(page);
  st = await state(page);
  const selAfter = await page.evaluate(() => window.__sarthink.selected >= 0 ? window.__sarthink.nodes[window.__sarthink.selected].id : null);
  check('refresh restores the selected node from the URL', selAfter === selId, `${selAfter} vs ${selId}`);
  await page.keyboard.press('Escape');
  await settle(page);

  check('no page errors in the main flow', page.errors.length === 0, page.errors.join(' | '));
  return page;
}

// Semantic search with a stubbed API response (deterministic) + selection/reset compatibility.
async function semanticFlow(browser) {
  let thread = null;
  const page = await open(browser, {
    route: async p => {
      await p.route('**/api/search', async r => {
        const body = JSON.parse(r.request().postData());
        const nodes = await p.evaluate(() => {
          const S = window.__sarthink;
          const t = S.nodes.filter(n => n.kind === 'thread' && n.links.length >= 2).slice(0, 2);
          return t.map(n => ({ id: n.id, title: n.title, people: n.links.map(k => { const e = S.edges[k]; return S.nodes[e.s === n.i ? e.t : e.s].label; }) }));
        });
        thread = nodes[0];
        const results = nodes.map((n, k) => ({
          rank: k + 1, similarity: 0.7 - k * 0.1, distance: 0.3 + k * 0.1, start_time: '2026-01-01T00:00:00+00:00', end_time: null,
          platform: 'reddit', title: n.title, channel_id: n.id.slice(2), node_id: n.id, people: n.people, summary: null,
          snippet: `talking about ${body.query} here`, text: `talking about ${body.query} here, in full`,
        }));
        results.push({ rank: 3, similarity: 0.4, distance: 0.6, start_time: null, end_time: null, platform: 'reddit', title: 'orphan',
          channel_id: null, node_id: null, people: [], summary: null, snippet: 'nothing', text: 'nothing' });
        await r.fulfill({ json: { query: body.query, table: 'topics', model: 'm', count: results.length, took_ms: 12, results } });
      });
    },
  });
  await ready(page);
  await page.fill('#sem-q', 'camera photography');
  await page.press('#sem-q', 'Enter');
  await page.waitForFunction(() => window.__sarthink.sem.set, null, { timeout: 10000 });
  await settle(page);
  let st = await state(page);
  check('memory search highlights threads + participants', st.semSet >= 2 && !st.semHidden, `${st.semSet} nodes`);
  const sub = await page.textContent('#sem-sub');
  check('results report how many are on the graph', /2 OF 3 ON GRAPH/.test(sub), sub);
  const why = await page.$$eval('#sem-list .sr-why', els => els.map(e => e.innerText));
  check('each result says why it matched', why.length === 3 && why.every(w => /similarity/.test(w)), why[0]);
  const marks = await page.$$eval('#sem-list .sr-snip mark', els => els.length);
  check('query words are marked in snippets', marks > 0);
  const semDim = await page.evaluate(() => { const S = window.__sarthink; return S.nodes.some((n, i) => !S.sem.set.has(i) && S.pAlpha[i] > 0 && S.pAlpha[i] < 0.5); });
  check('non-matching nodes are dimmed during memory search', semDim);
  await shot(page, '05_semantic');

  // Clicking a result selects its thread; semantic highlight stays.
  await page.click('#sem-list .sr >> nth=0');
  await settle(page);
  st = await state(page);
  const selId = await page.evaluate(() => window.__sarthink.selected >= 0 && window.__sarthink.nodes[window.__sarthink.selected].id);
  check('clicking a result selects its thread', selId === thread.id, `${selId} vs ${thread.id}`);
  const framed3 = await selectionInFreeArea(page);
  check('result flight frames thread + neighbours clear of both right panels', framed3.ok, `${framed3.bad}/${framed3.n} outside`);
  check('selection and memory highlight coexist', st.semSet >= 2 && st.focusEdges > 0);
  const whyPanel = await page.evaluate(() => { const w = document.querySelector('#details .why'); return w ? w.innerText : ''; });
  check('details explain why the node is highlighted', /Memory result #1/.test(whyPanel), whyPanel.slice(0, 90));
  await shot(page, '06_semantic_selected');

  // Selecting another node by clicking keeps the memory highlight.
  const other = await isolatedNode(page, 1);
  if (other) {
    await page.mouse.click(other.x, other.y);
    st = await state(page);
    check('canvas selection keeps memory highlight', st.selected === other.i && st.semSet >= 2);
  }

  // Clear (panel button) removes only the memory search; Esc resets everything.
  await page.click('#sem-panel .btn');
  st = await state(page);
  check('Clear removes memory highlight but keeps selection', st.semSet === 0 && st.semHidden && (other ? st.selected === other.i : true));
  await page.fill('#sem-q', 'camera');
  await page.press('#sem-q', 'Enter');
  await page.waitForFunction(() => window.__sarthink.sem.set);
  await page.keyboard.press('Escape');
  await settle(page);
  st = await state(page);
  check('Esc clears memory search, selection and camera', st.semSet === 0 && st.semHidden && st.selected === -1 && dist(st.cam, st.home) < 1);
  check('no page errors in the semantic flow', page.errors.length === 0, page.errors.join(' | '));
  await page.close();
}

// Ask Sarthink with a stubbed /api/ask (synthetic text only): tabs, chips, filters, rendering,
// escaping, graph focus from sources and timeline, index-building retry, and that the graph stays usable.
async function askFlow(browser) {
  const asks = [];
  let mode = 'ok';
  const page = await open(browser, {
    route: async p => {
      await p.route('**/api/ask', async r => {
        const body = JSON.parse(r.request().postData());
        asks.push(body);
        await new Promise(res => setTimeout(res, 800));   // long enough to see the loading state
        if (mode === 'building') {
          return r.fulfill({ status: 503, json: { error: { code: 'index_unavailable', message: "Table 'topics' not found" } } });
        }
        const nodes = await p.evaluate(() => window.__sarthink.nodes.filter(n => n.kind === 'thread' && n.links.length >= 2).slice(0, 2)
          .map(n => ({ id: n.id, title: n.title })));
        const evil = '<img src=x onerror="window.__xss=1">';
        const sources = [
          { rank: 1, node_id: nodes[0].id, title: nodes[0].title, platform: 'reddit', date_start: '2024-03-02T09:00:00+00:00',
            date_end: '2024-03-02T10:00:00+00:00', similarity: 0.71, snippet: `sourdough starter smells like apples ${evil}`,
            text: 'Full synthetic text about the sourdough starter.', people: [], relevant: true, matched_terms: ['sourdough'] },
          { rank: 2, node_id: nodes[1].id, title: nodes[1].title, platform: 'claude', date_start: '2024-01-10T09:00:00+00:00',
            date_end: '2024-01-10T09:30:00+00:00', similarity: 0.66, snippet: 'proofing sourdough in a cold kitchen',
            text: 'Synthetic chat about proofing.', people: [], relevant: true, matched_terms: ['sourdough'] },
          { rank: 3, node_id: null, title: 'orphan', platform: 'reddit', date_start: null, date_end: null, similarity: 0.4,
            snippet: 'unrelated lead', text: 'unrelated lead', people: [], relevant: false, matched_terms: [] },
        ];
        await r.fulfill({ json: {
          question: body.question, answer: `Found 2 memories that mention “sourdough”. ${evil}`, confidence: 'medium',
          summary_points: [`Jan 2024 · Claude: “proofing sourdough in a cold kitchen”`, `Mar 2024 · Reddit: “sourdough starter” ${evil}`],
          timeline: [
            { date: '2024-01-10T09:00:00+00:00', label: nodes[1].title, node_id: nodes[1].id, platform: 'claude', source: 2 },
            { date: '2024-03-02T09:00:00+00:00', label: `starter ${evil}`, node_id: nodes[0].id, platform: 'reddit', source: 1 },
          ],
          sources, notes: ['Merged 1 overlapping or duplicate chunk(s).'],
          evidence: { retrieved: 10, considered: 6, relevant: 2, terms: ['sourdough'] },
          filters: {}, model: 'm', took_ms: 42,
        } });
      });
    },
  });
  await ready(page);

  // Tabs: search stays the default and keeps working; Ask is one click away.
  check('memory search is the default tab', await page.isVisible('#sem-q') && !(await page.isVisible('#ask-q')));
  await page.click('#tab-ask');
  check('Ask tab shows the question box, chips and scope', await page.isVisible('#ask-q') && (await page.$$('#ask-chips .ask-chip')).length >= 3
    && /all platforms · all time/.test(await page.textContent('#ask-scope')));

  // Chip → request; loading state while it runs.
  await page.click('#ask-chips .ask-chip >> nth=0');
  const loading = await page.waitForFunction(() => document.querySelector('#sem-list .ask-loading') && document.getElementById('ask-go').disabled
    && document.querySelector('#ask-chips .ask-chip').disabled, null, { timeout: 2000 }).then(() => true, () => false);
  check('ask shows a loading state and disables the button', loading);
  await page.waitForSelector('#sem-list .ask-brief', { timeout: 10000 });
  await settle(page);
  check('chip submits its question', asks.length === 1 && asks[0].question === await page.inputValue('#ask-q') && !asks[0].platforms && !asks[0].date_from,
    JSON.stringify(asks[0]));
  const r = await page.evaluate(() => ({
    conf: document.querySelector('.ask-conf').className, answer: document.querySelector('.ask-answer').innerText,
    points: document.querySelectorAll('.ask-points li').length, tl: document.querySelectorAll('.atl-item').length,
    cards: document.querySelectorAll('#sem-list .sr').length, leads: document.querySelectorAll('#sem-list .sr.lead').length,
    imgs: document.querySelectorAll('#sem-list img').length, xss: window.__xss === 1,
  }));
  check('brief renders answer, confidence, key points, timeline and sources',
    /medium/.test(r.conf) && /sourdough/.test(r.answer) && r.points === 2 && r.tl === 2 && r.cards === 3 && r.leads === 1, JSON.stringify(r));
  check('archive text is escaped (no injected elements)', r.imgs === 0 && !r.xss && /<img src=x/.test(r.answer));
  let st = await state(page);
  check('ask sources highlight their threads on the graph', st.semSet >= 2 && !st.semHidden);
  await shot(page, '11_ask');

  // Timeline entry → selects that source's thread, like clicking a search result.
  const want = asks.length && await page.evaluate(() => window.__sarthink.sem.results[1].node_id);
  await page.click('.atl-item >> nth=0');
  await settle(page);
  let selId = await page.evaluate(() => window.__sarthink.selected >= 0 && window.__sarthink.nodes[window.__sarthink.selected].id);
  check('timeline item focuses its thread', selId === want, `${selId} vs ${want}`);
  check('timeline item marks its source card active', await page.evaluate(() => document.querySelectorAll('#sem-list .sr')[1].classList.contains('active')));
  const whyPanel = await page.evaluate(() => { const w = document.querySelector('#details .why'); return w ? w.innerText : ''; });
  check('details explain the Ask highlight', /Ask Sarthink source #2/.test(whyPanel), whyPanel.slice(0, 80));
  await page.click('#sem-list .sr >> nth=0');
  await settle(page);
  selId = await page.evaluate(() => window.__sarthink.nodes[window.__sarthink.selected].id);
  check('source card focuses its thread', selId === await page.evaluate(() => window.__sarthink.sem.results[0].node_id));
  const framed = await selectionInFreeArea(page);
  check('source flight frames the thread clear of the panels', framed.ok, `${framed.bad}/${framed.n} outside`);

  // The graph stays interactive with the brief open.
  const before = await state(page);
  await page.mouse.move(700, 450);
  await page.mouse.down();
  for (let k = 1; k <= 8; k++) await page.mouse.move(700 + k * 16, 450 + k * 5);
  await page.mouse.up();
  await settle(page);
  check('orbit still works with the Ask panel open', dist(before.cam, (await state(page)).cam) > 1);

  // Sidebar filters constrain the question: hide a platform, narrow the timeline, press Enter.
  const plats = await page.evaluate(() => window.__sarthink.platforms.map(p => p.key));
  await page.click(`#plat-list .frow[data-p="${plats[0]}"]`);
  const hasTime = await page.evaluate(() => !!window.__sarthink.time);
  if (hasTime) await page.click('#tl-presets button[data-m="12"]');
  check('scope line follows the filters', !/all platforms/.test(await page.textContent('#ask-scope')));
  await page.fill('#ask-q', 'what about <b>sourdough</b>?');
  await page.press('#ask-q', 'Enter');
  await page.waitForFunction(n => document.querySelector('.ask-brief') && !document.getElementById('ask-go').disabled, null, { timeout: 10000 });
  const last = asks[asks.length - 1];
  check('Enter submits with platform filter', last.question === 'what about <b>sourdough</b>?' && Array.isArray(last.platforms) && !last.platforms.includes(plats[0])
    && last.platforms.length === plats.length - 1, JSON.stringify(last.platforms));
  if (hasTime) check('Ask sends the timeline range', last.date_from && last.date_to && last.date_from < last.date_to, `${last.date_from} → ${last.date_to}`);
  check('question title is escaped', (await page.textContent('#sem-title')).includes('<b>sourdough</b>'));
  await page.fill('#ask-q', 'line one');
  await page.press('#ask-q', 'Shift+Enter');
  check('Shift+Enter adds a newline instead of submitting', (await page.inputValue('#ask-q')) === 'line one\n' && asks[asks.length - 1] === last);

  // Index still building → clear message + Retry that works once the index is ready.
  mode = 'building';
  await page.fill('#ask-q', 'sourdough');
  await page.click('#ask-go');
  await page.waitForSelector('#ask-retry', { timeout: 10000 });
  await shot(page, '12_ask_building');
  check('index-building state explains itself', /still building/i.test(await page.textContent('#sem-list')) && /INDEX STILL BUILDING/.test(await page.textContent('#sem-sub')));
  mode = 'ok';
  await page.click('#ask-retry');
  await page.waitForSelector('#sem-list .ask-brief', { timeout: 10000 });
  check('Retry re-asks and renders the brief', asks[asks.length - 1].question === 'sourdough');

  // Search tab still works alongside, and Esc resets everything.
  await page.click('#tab-search');
  check('switching back shows memory search', await page.isVisible('#sem-q') && !(await page.isVisible('#ask-q')));
  await page.keyboard.press('Escape');
  await settle(page);
  st = await state(page);
  check('Esc clears the Ask brief, highlight and filters', st.semSet === 0 && st.semHidden && st.selected === -1 && st.visibleNodes === st.nodes);
  check('no page errors in the Ask flow', page.errors.length === 0, page.errors.join(' | '));
  await page.close();

  // Narrow viewport: the Ask UI fits without horizontal scroll.
  const small = await open(browser, { viewport: { width: 1024, height: 700 } });
  await ready(small);
  await small.click('#tab-ask');
  const fit = await small.evaluate(() => {
    const side = document.getElementById('sidebar').getBoundingClientRect(), ask = document.getElementById('ask-go').getBoundingClientRect();
    return { docW: document.documentElement.scrollWidth <= innerWidth, btn: ask.right <= side.right && ask.bottom <= innerHeight };
  });
  check('1024x700: Ask controls fit in the sidebar, no horizontal scroll', fit.docW && fit.btn, JSON.stringify(fit));
  await small.close();
}

async function apiStates(browser) {
  // API offline
  let page = await open(browser, { route: p => p.route('**/api/**', r => r.abort()) });
  await ready(page);
  await page.waitForFunction(() => /offline/i.test(document.getElementById('sem-status').innerText));
  check('API offline state in the sidebar', true);
  await page.fill('#sem-q', 'hello');
  await page.press('#sem-q', 'Enter');
  await page.waitForFunction(() => /offline/i.test(document.getElementById('sem-list').innerText));
  check('API offline state in the results panel', true);
  const ti = await page.evaluate(() => window.__sarthink.nodes.findIndex(n => n.kind === 'thread'));
  await page.evaluate(i => { location.hash = '#node=' + window.__sarthink.nodes[i].id; }, ti);
  await page.waitForFunction(() => { const c = document.getElementById('d-ctx'); return c && /local API/.test(c.innerText); }, null, { timeout: 15000 });
  check('graph still fully usable with the API offline', (await state(page)).selected === ti);
  await page.close();

  // Index unavailable
  page = await open(browser, {
    route: async p => {
      await p.route('**/api/health', r => r.fulfill({ json: { status: 'ok', index: { table: 'topics', available: false, reason: "Table 'topics' not found" } } }));
      await p.route('**/api/search', r => r.fulfill({ status: 503, json: { error: { code: 'index_unavailable', message: "Table 'topics' not found" } } }));
    },
  });
  await ready(page);
  await page.waitForFunction(() => /not available yet/.test(document.getElementById('sem-status').innerText));
  await page.fill('#sem-q', 'hello');
  await page.press('#sem-q', 'Enter');
  await page.waitForFunction(() => /INDEX UNAVAILABLE/.test(document.getElementById('sem-sub').innerText));
  check('index-unavailable state', true);
  await page.close();
}

async function dataStates(browser) {
  const withCsv = (fn) => async p => {
    await p.route('**/cosmograph_nodes.csv', async r => {
      const res = await r.fetch();
      await r.fulfill({ response: res, body: fn(await res.text()) });
    });
  };
  const csvMap = (txt, rowFn) => {
    const lines = txt.trim().split(/\r?\n/);
    const head = lines[0].split('","').map(s => s.replace(/"/g, ''));
    const ix = c => head.indexOf(c);
    return [lines[0], ...lines.slice(1).map((l, k) => {
      const cells = l.slice(1, -1).split('","');
      rowFn(cells, ix, k);
      return '"' + cells.join('","') + '"';
    })].join('\n');
  };

  // Missing CSV
  let page = await open(browser, { route: p => p.route('**/cosmograph_nodes.csv', r => r.fulfill({ status: 404, body: 'nope' })) });
  await page.waitForFunction(() => document.body.dataset.graph === 'error');
  const msg = await page.textContent('#graph-note');
  check('missing CSV: Memory Home explains with the pipeline commands', /missing/.test(msg) && /export_cosmograph/.test(msg) && /compute_layout/.test(msg));
  check('missing CSV: no blocking overlay, question box usable', await page.isVisible('#home') && await page.isVisible('#omni-q') && !(await page.isVisible('#loading')));
  check('missing CSV: Explore graph disabled, status says unavailable', await page.isDisabled('#home-explore') && /unavailable/i.test(await page.textContent('#hs-graph')));
  await page.close();

  // Empty CSV
  page = await open(browser, { route: withCsv(t => t.split('\n')[0] + '\n') });
  await page.waitForFunction(() => /empty/.test(document.getElementById('graph-note').innerText));
  check('empty export shows an empty state', true);
  await page.close();

  // No layout columns → deterministic fallback
  const fallbackPositions = [];
  for (let run = 0; run < 2; run++) {
    page = await open(browser, { route: withCsv(t => csvMap(t, (c, ix) => { c[ix('layout_x')] = ''; c[ix('layout_y')] = ''; c[ix('layout_z')] = ''; })) });
    await ready(page);
    const s = await state(page);
    check(`missing layout → fallback layout (run ${run + 1})`, /fallback/.test(s.layoutNote), s.layoutNote);
    fallbackPositions.push(await page.evaluate(() => Array.from(window.__sarthink.pGeo.attributes.position.array.slice(0, 60))));
    if (run === 0) await shot(page, '07_fallback');
    await page.close();
  }
  check('fallback layout is deterministic', fallbackPositions[0].every((v, k) => v === fallbackPositions[1][k]));

  // All positions identical → fallback
  page = await open(browser, { route: withCsv(t => csvMap(t, (c, ix) => { c[ix('layout_x')] = '5'; c[ix('layout_y')] = '5'; c[ix('layout_z')] = '5'; })) });
  await ready(page);
  let s = await state(page);
  check('identical positions → fallback layout', /same layout position/.test(s.layoutNote), s.layoutNote);
  await page.close();

  // A few invalid rows → only those nodes are placed, the rest keep their exported positions
  page = await open(browser, { route: withCsv(t => csvMap(t, (c, ix, k) => { if (k % 50 === 0) c[ix('layout_y')] = k % 100 ? 'NaN' : ''; })) });
  await ready(page);
  s = await state(page);
  const kept = await page.evaluate(async () => {
    const S = window.__sarthink;
    const txt = await (await fetch('./processed_data/graph/cosmograph_nodes.csv', { cache: 'no-store' })).text();
    const rows = Papa.parse(txt, { header: true, skipEmptyLines: true }).data;
    let same = 0, moved = 0, finite = true;
    rows.forEach(r => {
      const n = S.byId.get(r.id);
      if (!Number.isFinite(n.x + n.y + n.z)) finite = false;
      const valid = r.layout_y !== '' && Number.isFinite(+r.layout_y);
      if (valid && n.x === +r.layout_x && n.y === +r.layout_y) same++; else if (!valid) moved++;
    });
    return { same, moved, finite, rows: rows.length };
  });
  check('invalid rows get fallback positions, valid rows keep theirs', /invalid positions/.test(s.layoutNote) && kept.finite && kept.same + kept.moved === kept.rows,
    `${kept.moved} placed, ${kept.same} kept`);
  await page.close();

  // A malformed or unknown #node= link is ignored instead of breaking the page
  for (const h of ['#node=%E0%A4', '#node=T_does_not_exist']) {
    page = await open(browser, { url: BASE + h });
    await ready(page);
    s = await state(page);
    check(`bad link ${h} is ignored`, s.selected === -1 && page.errors.length === 0, page.errors.join(' | '));
    await page.close();
  }

  // Old export without date columns → timeline explains itself
  page = await open(browser, { route: withCsv(t => csvMap(t, (c, ix) => { c[ix('first_ts')] = ''; c[ix('last_ts')] = ''; })) });
  await ready(page);
  const tl = await page.textContent('#timeline');
  check('timeline explains when dates are missing', /no activity dates/.test(tl));
  await page.close();
}

async function responsive(browser) {
  for (const vp of [{ width: 1280, height: 720 }, { width: 1366, height: 768 }, { width: 1024, height: 700 }]) {
    const page = await open(browser, { viewport: vp });
    await ready(page);
    const overflow = await page.evaluate(() => {
      const side = document.getElementById('sidebar').getBoundingClientRect();
      return { side: side.bottom <= innerHeight + 1, docW: document.documentElement.scrollWidth <= innerWidth };
    });
    const n = await isolatedNode(page);
    if (n) await page.mouse.click(n.x, n.y);
    const panels = await page.evaluate(() => {
      const a = document.getElementById('sidebar').getBoundingClientRect(), b = document.getElementById('details').getBoundingClientRect();
      return { overlap: !(a.right <= b.left || b.right <= a.left), fits: b.bottom <= innerHeight + 1 };
    });
    check(`${vp.width}x${vp.height}: sidebar fits, no horizontal scroll`, overflow.side && overflow.docW);
    check(`${vp.width}x${vp.height}: details panel fits beside sidebar`, !panels.overlap && panels.fits);
    await shot(page, `08_${vp.width}x${vp.height}`);
    await page.close();
  }
}

// ─── Memory Home, Insights and arrow keys (all API answers synthetic) ─────────────
const EVIL = '<img src=x onerror="window.__xss=1">';

// Thread nodes (ids only) the stubs can cite, read from the page's own graph.
function graphThreads(page, n = 2) {
  return page.evaluate(n => window.__sarthink.nodes.filter(x => x.kind === 'thread' && x.links.length >= 2).slice(0, n).map(x => x.id), n);
}

async function stubAsk(p, calls) {
  await p.route('**/api/ask', async r => {
    const body = JSON.parse(r.request().postData());
    calls.push({ kind: 'ask', ...body });
    await new Promise(res => setTimeout(res, 300));
    const ids = await graphThreads(p).catch(() => []);
    const src = (k, id) => ({ rank: k + 1, node_id: id || null, title: `Synthetic thread ${k + 1} ${EVIL}`, platform: k ? 'claude' : 'reddit',
      date_start: `2024-0${k + 1}-02T09:00:00+00:00`, date_end: null, similarity: 0.7 - k / 10, snippet: `synthetic sourdough note ${k} ${EVIL}`,
      text: 'Synthetic full text about sourdough.', people: [], relevant: true, matched_terms: ['sourdough'] });
    const sources = [src(0, ids[0]), src(1, ids[1])];
    await r.fulfill({ json: {
      question: body.question, answer: `Found 2 memories that mention “sourdough”. ${EVIL}`, confidence: 'medium',
      summary_points: [`Jan 2024 · Reddit: “sourdough” ${EVIL}`], timeline: sources.map((s, k) => ({ date: s.date_start, label: s.title, node_id: s.node_id, platform: s.platform, source: k + 1 })),
      sources, notes: [], evidence: { retrieved: 5, considered: 4, relevant: 2, terms: ['sourdough'] }, filters: {}, model: 'm', took_ms: 21,
    } });
  });
  await p.route('**/api/search', async r => {
    const body = JSON.parse(r.request().postData());
    calls.push({ kind: 'search', ...body });
    const ids = await graphThreads(p).catch(() => []);
    const results = ids.map((id, k) => ({ rank: k + 1, similarity: 0.6, distance: 0.4, start_time: '2025-01-01T00:00:00+00:00', end_time: null, platform: 'reddit',
      title: `Synthetic result ${k + 1}`, channel_id: id.slice(2), node_id: id, people: [], summary: null, snippet: `about ${body.query}`, text: `about ${body.query}` }));
    await r.fulfill({ json: { query: body.query, table: 'topics', model: 'm', count: results.length, took_ms: 9, results } });
  });
}

// A synthetic /api/insights answer whose contacts/conversations point at real node ids (so focus works).
async function fakeInsights(p, url, { empty = false } = {}) {
  const q = new URL(url).searchParams;
  const ids = await p.evaluate(() => {
    const S = window.__sarthink;
    if (!S.nodes.length) return { users: ['U_999999'], threads: ['T_999999'] };
    return { users: S.nodes.filter(n => n.kind === 'user').slice(0, 3).map(n => n.id), threads: S.nodes.filter(n => n.kind === 'thread').slice(0, 3).map(n => n.id) };
  });
  const months = [];
  for (let y = 2023; y <= 2024; y++) for (let m = 1; m <= 12; m++) {
    const reddit = (y - 2022) * 10 + m, insta = m % 3 ? 4 : 0;
    months.push({ month: `${y}-${String(m).padStart(2, '0')}`, total: reddit + insta, platforms: insta ? { reddit, instagram: insta } : { reddit } });
  }
  const total = months.reduce((a, m) => a + m.total, 0);
  const plats = q.get('platforms') ? q.get('platforms').split(',') : null;
  if (empty) {
    return { empty: true, filters: { platforms: plats, date_from: q.get('date_from'), date_to: q.get('date_to') }, totals: { messages: 0, threads: 0, people: 0, platforms: 0 },
      first_date: null, latest_date: null, platforms: [], available_platforms: ['instagram', 'reddit'], activity: { months: [], years: [] },
      top_contacts: [], top_conversations: [], owner: { configured: true, excluded_accounts: 2 }, took_ms: 3, cached: false };
  }
  return {
    empty: false, filters: { platforms: plats, date_from: q.get('date_from'), date_to: q.get('date_to') },
    totals: { messages: total, threads: 12, people: 3, platforms: 2 }, first_date: '2023-01-03T10:00:00+00:00', latest_date: '2024-12-20T10:00:00+00:00',
    platforms: [
      { platform: 'reddit', messages: total - 64, threads: 9, people: 2, first_date: '2023-01-03T10:00:00+00:00', last_date: '2024-12-20T10:00:00+00:00', share: 0.9 },
      { platform: 'instagram', messages: 64, threads: 3, people: 1, first_date: '2023-01-05T10:00:00+00:00', last_date: '2024-11-02T10:00:00+00:00', share: 0.1 },
    ],
    available_platforms: ['instagram', 'reddit'],
    activity: { months, years: [{ year: 2023, total: 200, platforms: {} }, { year: 2024, total: total - 200, platforms: {} }] },
    top_contacts: ids.users.map((id, k) => ({ node_id: id, label: k ? `Synthetic Friend ${k}` : `Synthetic ${EVIL}`, platform: 'reddit', messages: 90 - k * 10, threads: 2,
      first_date: '2023-02-01T00:00:00+00:00', last_date: '2024-02-01T00:00:00+00:00' })),
    top_conversations: ids.threads.map((id, k) => ({ node_id: id, title: `Synthetic conversation ${k + 1} ${EVIL}`, platform: 'reddit', messages: 300 - k * 50, people: 3,
      first_date: '2023-03-01T00:00:00+00:00', last_date: '2024-03-01T00:00:00+00:00' })),
    owner: { configured: true, excluded_accounts: 2 }, took_ms: 12, cached: false,
  };
}

async function homeFlow(browser) {
  const calls = [];
  const page = await open(browser, { route: p => stubAsk(p, calls) });
  await ready(page, { view: 'home' });
  await shot(page, '20_home');

  // Opening state: Memory Home in front, graph behind, workspace panels out of the way.
  const open0 = await page.evaluate(() => ({
    view: document.body.dataset.view, askMode: document.getElementById('omode-ask').getAttribute('aria-selected') === 'true',
    chips: document.querySelectorAll('#home-chips .ask-chip').length, sidebar: getComputedStyle(document.getElementById('sidebar')).visibility,
    status: document.getElementById('home-status').innerText,
  }));
  check('opens on Memory Home with Ask as the default', open0.view === 'home' && open0.askMode && await page.isVisible('#omni-q'));
  check('Memory Home shows sample prompts and a status line', open0.chips >= 3 && /Graph/.test(open0.status) && /Memory index|API/.test(open0.status), `${open0.chips} chips`);
  check('graph workspace panels are hidden on Memory Home', open0.sidebar === 'hidden' && !(await page.isVisible('#sem-q')) && !(await page.isVisible('#find-q')));
  const drawn = await page.waitForFunction(() => getComputedStyle(document.getElementById('stage')).opacity === '1', null, { timeout: 3000 }).then(() => true, () => false);
  check('graph is drawn behind Memory Home', drawn && await page.evaluate(() => window.__sarthink.visibleNodes > 0));

  // The graph stays interactive outside the card.
  const outside = await page.evaluate(() => {
    const h = document.getElementById('home').getBoundingClientRect(), canvas = document.querySelector('#stage canvas');
    const pts = [[40, innerHeight - 40], [innerWidth - 40, innerHeight / 2], [h.left + 20, h.bottom + 30]];
    return { canvas: pts.every(([x, y]) => document.elementFromPoint(x, y) === canvas), card: document.getElementById('home').contains(document.elementFromPoint(h.left + 30, h.top + 30)) };
  });
  check('pointer: canvas outside the Home card, card inside it', outside.canvas && outside.card);
  let b = await state(page);
  await page.mouse.move(80, 820);
  await page.mouse.down();
  for (let k = 1; k <= 8; k++) await page.mouse.move(80 + k * 20, 820 - k * 4);
  await page.mouse.up();
  await page.waitForTimeout(400);
  let a = await state(page);
  check('orbit works on Memory Home (outside the card)', dist(b.cam, a.cam) > 1);
  b = a;
  await page.mouse.move(120, 800);
  for (let k = 0; k < 4; k++) { await page.mouse.wheel(0, -200); await page.waitForTimeout(40); }
  await page.waitForTimeout(500);
  a = await state(page);
  check('wheel zoom works on Memory Home', dist(a.cam, a.target) < dist(b.cam, b.target) * 0.97);

  // Ask from a sample chip: the brief renders inside Memory Home, escaped.
  await page.click('#home-chips .ask-chip >> nth=0');
  await page.waitForSelector('#home .ask-brief', { timeout: 10000 });
  const brief = await page.evaluate(() => ({
    inHome: document.getElementById('home').contains(document.getElementById('sem-panel')), cls: document.body.classList.contains('has-results'),
    cards: document.querySelectorAll('#home #sem-list .sr').length, imgs: document.querySelectorAll('#home img, #sem-list img').length, xss: window.__xss === 1,
    answer: document.querySelector('.ask-answer').innerText,
  }));
  check('Ask from Memory Home renders the brief in place', brief.inHome && brief.cls && brief.cards === 2 && calls.at(-1).kind === 'ask');
  check('Memory Home: archive text is escaped', brief.imgs === 0 && !brief.xss && brief.answer.includes('<img src=x'));
  check('Ask stays on Memory Home until a source is chosen', (await state(page)).selected === -1 && await page.evaluate(() => document.body.dataset.view === 'home'));
  await shot(page, '21_home_ask');

  // Switch to Find memories: placeholder, chips and endpoint change.
  await page.click('#omode-search');
  const mode = await page.evaluate(() => ({ sel: document.getElementById('omode-search').getAttribute('aria-selected'), ph: document.getElementById('omni-q').placeholder, go: document.getElementById('omni-go').textContent }));
  check('Find memories mode switches the box', mode.sel === 'true' && /memory/i.test(mode.ph) && mode.go === 'Search');
  await page.fill('#omni-q', 'synthetic cameras');
  await page.press('#omni-q', 'Enter');
  await page.waitForFunction(() => /Synthetic result/.test(document.getElementById('sem-list').innerText), null, { timeout: 10000 });
  check('Find memories runs a semantic search', calls.at(-1).kind === 'search' && calls.at(-1).query === 'synthetic cameras');
  check('sidebar search box stays in sync', await page.inputValue('#sem-q') === 'synthetic cameras');
  await page.keyboard.press('ArrowLeft');   // focus is in the textarea: must not move anything
  await page.click('#omode-ask');

  // Ask again, then click a cited source → graph workspace focused on its thread.
  await page.fill('#omni-q', 'sourdough?');
  await page.press('#omni-q', 'Enter');
  await page.waitForSelector('#home .ask-brief', { timeout: 10000 });
  await page.waitForFunction(() => !document.getElementById('omni-go').disabled);
  const want = await page.evaluate(() => window.__sarthink.sem.results[0].node_id);
  await page.click('#home #sem-list .sr >> nth=0');
  await settle(page);
  let st = await state(page);
  const after = await page.evaluate(() => ({ view: document.body.dataset.view, panelInRight: document.getElementById('rightcol').contains(document.getElementById('sem-panel')),
    omniInTop: document.getElementById('topbar').contains(document.getElementById('omni')) }));
  check('cited source opens the graph workspace', after.view === 'graph' && after.panelInRight && after.omniInTop);
  check('cited source selects and highlights its thread', st.selected >= 0 && await page.evaluate(() => window.__sarthink.nodes[window.__sarthink.selected].id) === want && st.semSet >= 2);
  const framed = await selectionInFreeArea(page);
  check('cited source is framed clear of the panels', framed.ok, `${framed.bad}/${framed.n} outside`);
  await shot(page, '22_source_on_graph');

  // Command bar in the workspace.
  await page.fill('#omni-q', 'sourdough again');
  await page.press('#omni-q', 'Enter');
  await page.waitForFunction(() => !document.getElementById('omni-go').disabled && document.querySelector('#rightcol .ask-brief'), null, { timeout: 10000 });
  check('command bar asks from the graph workspace', calls.at(-1).question === 'sourdough again' && await page.evaluate(() => document.body.dataset.view === 'graph'));
  check('graph clicks still work with the command bar', (await state(page)).focusEdges > 0);

  // G toggles back to Home (results follow), and again to the graph.
  await page.click('#btn-help'); await page.click('#help-close');
  await page.evaluate(() => document.activeElement.blur());
  await page.keyboard.press('g');
  await page.waitForTimeout(500);
  check('G returns to Memory Home with the results', await page.evaluate(() => document.body.dataset.view === 'home' && document.getElementById('home').contains(document.getElementById('sem-panel'))));
  await page.keyboard.press('g');
  await settle(page);
  check('G opens the workspace again, keeping the selection', await page.evaluate(() => document.body.dataset.view === 'graph') && (await state(page)).selected >= 0);
  await page.keyboard.press('Escape');
  await settle(page);
  st = await state(page);
  check('Esc in the workspace still resets everything', st.selected === -1 && st.semSet === 0 && dist(st.cam, st.home) < 1);
  check('no page errors in the Memory Home flow', page.errors.length === 0, page.errors.join(' | '));
  await page.close();

  // Explore graph from a fresh page: smooth hand-over to a fully framed workspace.
  const p2 = await open(browser);
  await ready(p2, { view: 'home' });
  await p2.click('#home-explore');
  await p2.waitForFunction(() => getComputedStyle(document.getElementById('sidebar')).visibility === 'visible' && getComputedStyle(document.getElementById('home')).visibility === 'hidden', null, { timeout: 3000 });
  await settle(p2);
  const offscreen = await p2.evaluate(() => {
    const S = window.__sarthink, cam = S.camera; cam.updateMatrixWorld();
    const p = cam.projectionMatrix.elements, v = cam.matrixWorldInverse.elements, side = document.getElementById('sidebar').getBoundingClientRect();
    let out = 0;
    for (const n of S.nodes) {
      const mv = [0, 1, 2, 3].map(r => v[r] * n.x + v[4 + r] * n.y + v[8 + r] * n.z + v[12 + r]);
      const c = [0, 1, 2, 3].map(r => p[r] * mv[0] + p[4 + r] * mv[1] + p[8 + r] * mv[2] + p[12 + r] * mv[3]);
      const x = (c[0] / c[3] + 1) / 2 * innerWidth;
      if (c[3] <= 0 || x < side.right - 2 || Math.abs(c[0] / c[3]) > 1 || Math.abs(c[1] / c[3]) > 1) out++;
    }
    return out;
  });
  check('Explore graph: workspace opens with the whole graph framed beside the sidebar', offscreen === 0, `${offscreen} nodes hidden`);
  check('Explore graph: no auto-rotation in the workspace', await p2.evaluate(() => !window.__sarthink.controls.autoRotate));
  check('no page errors after Explore graph', p2.errors.length === 0, p2.errors.join(' | '));
  await p2.close();
}

async function arrowKeys(browser) {
  const page = await open(browser);
  await ready(page);
  const target = await isolatedNode(page);
  await page.mouse.click(Math.round(await page.evaluate(() => innerWidth)) - 420, 860);   // focus the canvas (empty space)
  const sp = async () => screenOf(page, target.i);
  for (const [key, axis, sign] of [['ArrowRight', 'x', 1], ['ArrowLeft', 'x', -1], ['ArrowUp', 'y', -1], ['ArrowDown', 'y', 1]]) {
    const b = await sp();
    for (let k = 0; k < 3; k++) await page.keyboard.press(key);
    await page.waitForTimeout(300);
    const a = await sp();
    const moved = (a[axis] - b[axis]) * sign, other = Math.abs(axis === 'x' ? a.y - b.y : a.x - b.x);
    check(`${key} moves the graph ${key.slice(5).toLowerCase()}`, moved > 20 && other < moved * 0.25, `${moved.toFixed(0)}px along, ${other.toFixed(0)}px across`);
  }
  let b = await state(page);
  await page.keyboard.press('Shift+ArrowLeft');
  await page.keyboard.press('Shift+ArrowLeft');
  await page.waitForTimeout(300);
  let a = await state(page);
  check('Shift+arrow orbits around the target', dist(a.cam, b.cam) > 1 && dist(a.target, b.target) < 1e-6);
  // Arrows keep their normal meaning in inputs and tab strips.
  await settle(page);
  b = await state(page);
  await page.focus('#find-q');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowDown');
  await page.focus('#tab-search');
  await page.keyboard.press('ArrowRight');
  a = await state(page);
  check('arrows in inputs and tabs do not move the graph', dist(a.cam, b.cam) < 1e-3 && dist(a.target, b.target) < 1e-3, `moved ${dist(a.cam, b.cam).toExponential(1)}`);
  check('arrows still switch the memory tabs', await page.evaluate(() => document.getElementById('tab-ask').getAttribute('aria-selected') === 'true'));
  check('no page errors with arrow keys', page.errors.length === 0, page.errors.join(' | '));
  await page.close();
}

async function insightsFlow(browser) {
  const reqs = [];
  let mode = 'ok';
  const page = await open(browser, {
    route: p => p.route('**/api/insights**', async r => {
      reqs.push(r.request().url());
      if (mode === 'offline') return r.abort();
      if (mode === 'db') return r.fulfill({ status: 503, json: { error: { code: 'database_unavailable', message: 'Memory database not found: x.db.' } } });
      await new Promise(res => setTimeout(res, 250));
      await r.fulfill({ json: await fakeInsights(p, r.request().url(), { empty: mode === 'empty' }) });
    }),
  });
  await ready(page, { view: 'home' });

  await page.click('#home-insights');
  await page.waitForSelector('#ins-chart', { timeout: 10000 });
  await page.waitForFunction(() => getComputedStyle(document.getElementById('home')).visibility === 'hidden', null, { timeout: 3000 }).catch(() => {});
  await shot(page, '30_insights_home');
  const r = await page.evaluate(() => ({
    cards: document.querySelectorAll('#insights .card').length, first: document.querySelector('#insights .card .v').innerText,
    bars: document.querySelectorAll('#ins-chart .col').length, plats: document.querySelectorAll('#insights .prow').length,
    contacts: document.querySelectorAll('#insights .ritem').length, imgs: document.querySelectorAll('#insights img').length, xss: window.__xss === 1,
    homeHidden: getComputedStyle(document.getElementById('home')).visibility === 'hidden', scope: document.getElementById('ins-scope').innerText,
    text: document.getElementById('ins-body').innerText,
  }));
  check('Insights opens from Memory Home with stat cards', r.cards === 5 && /\d/.test(r.first) && r.homeHidden, JSON.stringify({ cards: r.cards }));
  check('Insights draws one SVG bar column per month', r.bars === 24);
  check('Insights lists platforms, contacts and conversations', r.plats === 2 && r.contacts === 6 && /excluding you/.test(r.text));
  check('Insights escapes archive strings', r.imgs === 0 && !r.xss && r.text.includes('<img src=x'));
  check('Insights request is unfiltered by default', !new URL(reqs[0]).search && /All platforms · all time/.test(r.scope), reqs[0]);

  // Hover tooltip with the platform split.
  const col = page.locator('#ins-chart .col').nth(1);
  await col.hover();
  const tip = await page.evaluate(() => ({ show: document.getElementById('ins-tip').classList.contains('show'), text: document.getElementById('ins-tip').innerText }));
  check('chart hover shows month, total and platform split', tip.show && /Feb 2023/.test(tip.text) && /Reddit/.test(tip.text) && /Instagram/.test(tip.text), tip.text.replace(/\s+/g, ' '));

  // Contact → graph workspace, node selected.
  const contactId = await page.getAttribute('#insights .ritem >> nth=1', 'data-node');
  await page.click('#insights .ritem >> nth=1');
  await settle(page);
  let st = await state(page);
  check('clicking a contact closes Insights and focuses it on the graph',
    await page.evaluate(() => document.getElementById('insights').hidden && document.body.dataset.view === 'graph')
    && await page.evaluate(() => window.__sarthink.nodes[window.__sarthink.selected].id) === contactId);

  // Filters in the workspace flow into the request.
  await page.keyboard.press('Escape');
  await settle(page);
  const plats = await page.evaluate(() => window.__sarthink.platforms.map(p => p.key));
  await page.click(`#plat-list .frow[data-p="${plats[0]}"]`);
  const hasTime = await page.evaluate(() => !!window.__sarthink.time);
  if (hasTime) await page.click('#tl-presets button[data-m="12"]');
  await page.evaluate(() => document.activeElement.blur());
  await page.keyboard.press('i');
  await page.waitForFunction(() => !document.getElementById('insights').hidden && document.querySelector('#ins-chart'));
  await page.waitForTimeout(400);
  let q = new URL(reqs.at(-1)).searchParams;
  check('Insights respects the sidebar platform filter', q.get('platforms') && !q.get('platforms').split(',').includes(plats[0]), q.get('platforms'));
  if (hasTime) check('Insights respects the timeline range', q.get('date_from') && q.get('date_to') && q.get('date_from') < q.get('date_to'));
  check('Insights scope line shows the filters', /Clear filters/.test(await page.textContent('#ins-scope')));
  const sideVisible = await page.evaluate(() => {
    const i = document.getElementById('insights').getBoundingClientRect(), s = document.getElementById('sidebar').getBoundingClientRect();
    return i.left >= s.right;
  });
  check('in the workspace Insights sits beside the sidebar', sideVisible);
  await shot(page, '31_insights_graph');

  // Changing a filter while Insights is open refetches.
  const n0 = reqs.length;
  await page.click(`#plat-list .frow[data-p="${plats[0]}"]`);
  await page.waitForFunction(n => window.__reqsDone || true, n0);
  await page.waitForTimeout(700);
  q = new URL(reqs.at(-1)).searchParams;
  check('toggling a platform with Insights open refetches', reqs.length > n0 && !q.get('platforms'), q.get('platforms') || '(all)');

  // Empty, database-unavailable and offline states.
  mode = 'empty';
  await page.click('#insights .pchip >> nth=0');
  await page.waitForFunction(() => /No memories match/.test(document.getElementById('ins-body').innerText), null, { timeout: 5000 });
  check('Insights empty state offers to clear filters', await page.isVisible('#ins-act'));
  mode = 'db';
  await page.click('#ins-act');
  await page.waitForFunction(() => /database isn’t available/.test(document.getElementById('ins-body').innerText), null, { timeout: 5000 });
  check('Insights database-unavailable state', true);
  mode = 'offline';
  await page.click('#ins-act');
  await page.waitForFunction(() => /offline/.test(document.getElementById('ins-body').innerText), null, { timeout: 5000 });
  check('Insights API-offline state shows the start command', /start_sarthink/.test(await page.textContent('#ins-body')));
  mode = 'ok';
  await page.click('#ins-act');
  await page.waitForSelector('#ins-chart', { timeout: 5000 });
  check('Retry recovers Insights', true);
  await page.keyboard.press('Escape');
  check('Esc closes Insights first (workspace kept)', await page.evaluate(() => document.getElementById('insights').hidden && document.body.dataset.view === 'graph'));
  check('no page errors in the Insights flow', page.errors.length === 0, page.errors.join(' | '));
  await page.close();

  // Loading state is visible while the request runs.
  const slow = await open(browser, { route: p => p.route('**/api/insights**', async r => { await new Promise(res => setTimeout(res, 1500)); await r.fulfill({ json: await fakeInsights(p, r.request().url()) }); }) });
  await ready(slow, { view: 'home' });
  await slow.click('#home-insights');
  check('Insights shows a loading skeleton', await slow.waitForSelector('#insights .skel', { timeout: 1000 }).then(() => true, () => false));
  await slow.close();
}

async function homeWithoutGraph(browser) {
  const calls = [];
  const page = await open(browser, { route: async p => {
    await p.route('**/cosmograph_edges.csv', r => r.fulfill({ status: 404, body: 'nope' }));
    await stubAsk(p, calls);
    await p.route('**/api/insights**', async r => r.fulfill({ json: await fakeInsights(p, r.request().url()) }));
  } });
  await page.waitForFunction(() => document.body.dataset.graph === 'error');
  await page.fill('#omni-q', 'sourdough?');
  await page.press('#omni-q', 'Enter');
  await page.waitForSelector('#home .ask-brief', { timeout: 10000 });
  check('no graph: Ask still answers on Memory Home', calls.length === 1);
  await page.click('#home #sem-list .sr >> nth=0');
  check('no graph: a source expands in place instead of opening the graph', await page.evaluate(() => document.body.dataset.view === 'home' && document.querySelector('#sem-list .sr.expanded') !== null));
  await page.keyboard.press('g');
  check('no graph: G stays on Memory Home', await page.evaluate(() => document.body.dataset.view === 'home'));
  await page.click('#home-insights');
  await page.waitForSelector('#ins-chart', { timeout: 10000 });
  await page.click('#insights .ritem >> nth=0');
  check('no graph: Insights explains it cannot focus the graph', /graph isn’t available/.test(await page.textContent('#ins-toast')));
  await page.click('#insights .pchip >> nth=0');
  await page.waitForTimeout(400);
  check('no graph: Insights platform chips still filter', await page.evaluate(() => document.querySelector('#insights .pchip').getAttribute('aria-pressed') === 'false'));
  await shot(page, '32_no_graph');
  check('no page errors without the graph', page.errors.length === 0, page.errors.join(' | '));
  await page.close();
}

async function homeResponsive(browser) {
  for (const vp of [{ width: 1440, height: 900 }, { width: 1024, height: 700 }, { width: 390, height: 844 }]) {
    const page = await open(browser, { viewport: vp, route: p => p.route('**/api/insights**', async r => r.fulfill({ json: await fakeInsights(p, r.request().url()) })) });
    await ready(page, { view: 'home' });
    const fit = await page.evaluate(() => {
      const h = document.getElementById('home').getBoundingClientRect(), go = document.getElementById('omni-go').getBoundingClientRect();
      const tb = [...document.querySelectorAll('#tbtns .btn')].filter(b => b.offsetParent).map(b => b.getBoundingClientRect());
      return { docW: document.documentElement.scrollWidth <= innerWidth, card: h.left >= 0 && h.right <= innerWidth && h.bottom <= innerHeight + 1,
        go: go.right <= h.right && go.bottom <= innerHeight, buttons: tb.every(r => r.right <= innerWidth && r.left >= 0), overlap: tb.some(r => r.bottom > h.top) };
    });
    check(`${vp.width}x${vp.height}: Memory Home fits, no horizontal scroll`, fit.docW && fit.card && fit.go && fit.buttons && !fit.overlap, JSON.stringify(fit));
    await page.click('#home-insights');
    await page.waitForSelector('#ins-chart');
    const ins = await page.evaluate(() => {
      const i = document.getElementById('insights').getBoundingClientRect();
      return { docW: document.documentElement.scrollWidth <= innerWidth, box: i.left >= 0 && i.right <= innerWidth + 1 && i.bottom <= innerHeight + 1,
        cards: [...document.querySelectorAll('#insights .card')].every(c => c.getBoundingClientRect().right <= i.right) };
    });
    check(`${vp.width}x${vp.height}: Insights fits`, ins.docW && ins.box && ins.cards, JSON.stringify(ins));
    await shot(page, `33_insights_${vp.width}`);
    await page.keyboard.press('Escape');
    if (vp.width >= 1024) {
      await page.click('#home-explore');
      await settle(page);
      const ws = await page.evaluate(() => {
        const o = document.getElementById('omni').getBoundingClientRect(), s = document.getElementById('sidebar').getBoundingClientRect();
        const btns = [...document.querySelectorAll('#tbtns .btn')].filter(b => b.offsetParent).map(b => b.getBoundingClientRect());
        return { docW: document.documentElement.scrollWidth <= innerWidth, bar: o.width >= 200 && o.bottom <= s.top, clash: btns.some(b => b.left < o.right && b.right > o.left) };
      });
      check(`${vp.width}x${vp.height}: command bar fits the workspace top bar`, ws.docW && ws.bar && !ws.clash, JSON.stringify(ws));
    }
    await page.close();
  }
}

async function realSemantic(browser) {
  const page = await open(browser);
  await ready(page);
  await page.fill('#sem-q', 'college exams stress');
  await page.press('#sem-q', 'Enter');
  await page.waitForFunction(() => !document.getElementById('sem-go').disabled, null, { timeout: 200000 });
  const st = await state(page);
  const status = await page.textContent('#sem-status');
  check('real memory search returns results on the graph', st.semSet > 0, status.trim());
  await settle(page);
  await shot(page, '09_real_semantic');
  if (st.semSet) {
    await page.click('#sem-list .sr:not(.nonode) >> nth=0');
    await settle(page);
    const s2 = await state(page);
    check('real result selects its thread', s2.selected >= 0 && s2.semSet > 0);
    const f = await selectionInFreeArea(page);
    check('real result: thread + neighbours visible, not under panels', f.ok, `${f.bad}/${f.n} outside`);
    await page.waitForFunction(() => { const c = document.getElementById('d-ctx'); return c && !/Loading/.test(c.innerText); }, null, { timeout: 15000 });
    await shot(page, '10_real_semantic_selected');
  }
  await page.close();
}

// SARTHINK_E2E_ONLY=homeFlow,insightsFlow runs just those flows.
const ONLY = (process.env.SARTHINK_E2E_ONLY || '').split(',').filter(Boolean);
const FLOWS = { mainFlow: async b => (await mainFlow(b)).close(), semanticFlow, askFlow, apiStates, dataStates, responsive,
  homeFlow, arrowKeys, insightsFlow, homeWithoutGraph, homeResponsive, ...(REAL_SEMANTIC ? { realSemantic } : {}) };
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
try {
  for (const [name, flow] of Object.entries(FLOWS)) if (!ONLY.length || ONLY.includes(name)) await flow(browser);
} catch (e) {
  check('e2e run completed', false, e.stack || e.message);
} finally {
  await browser.close();
}
console.log(results.join('\n'));
console.log(`\n${results.length - failures}/${results.length} checks passed`);
process.exit(failures ? 1 : 0);
