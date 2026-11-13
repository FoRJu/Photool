import { test } from "node:test";
import assert from "node:assert/strict";
import { chromaOf, cellOf, scopeOf, SKIN_ANGLE, TARGETS } from "../src/vectorscope.js";

const angle = ([cb, cr]) => ((Math.atan2(cr, cb) * 180) / Math.PI + 360) % 360;
const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

test("a grey of any brightness is at the centre", () => {
  for (const v of [0, 1, 64, 128, 200, 255]) {
    const [cb, cr] = chromaOf(v, v, v);
    assert.ok(near(cb, 0) && near(cr, 0), `grey ${v} at ${cb}, ${cr}`);
  }
});

test("the primaries lie within the unit square", () => {
  for (const rgb of [[255, 0, 0], [0, 255, 0], [0, 0, 255]]) {
    const [cb, cr] = chromaOf(...rgb);
    assert.ok(Math.abs(cb) <= 1 + 1e-9 && Math.abs(cr) <= 1 + 1e-9, `${rgb} at ${cb}, ${cr}`);
  }
  assert.ok(chromaOf(255, 0, 0)[1] > 0, "red's cr is positive");
  assert.ok(chromaOf(0, 0, 255)[0] > 0, "blue's cb is positive");
});

test("the targets go round clockwise R, Mg, B, Cy, G, Yl", () => {
  assert.deepEqual(TARGETS.map((t) => t.label), ["R", "Mg", "B", "Cy", "G", "Yl"]);
  let total = 0;
  for (let i = 0; i < TARGETS.length; i++) {
    const a = angle(TARGETS[i].at);
    const b = angle(TARGETS[(i + 1) % TARGETS.length].at);
    const step = (a - b + 360) % 360; // clockwise: the angle falls
    assert.ok(step > 0 && step < 180, `${TARGETS[i].label} to the next is ${step}`);
    total += step;
  }
  assert.ok(near(total, 360, 1e-6), `once round, ${total}`);
});

test("the skin line lies between yellow and red", () => {
  const yl = angle(TARGETS.find((t) => t.label === "Yl").at);
  const r = angle(TARGETS.find((t) => t.label === "R").at);
  assert.ok(SKIN_ANGLE > r && SKIN_ANGLE < yl, `${r} < ${SKIN_ANGLE} < ${yl}`);
});

test("a typical skin colour is near the skin line", () => {
  const a = angle(chromaOf(224, 172, 150));
  assert.ok(Math.abs(a - SKIN_ANGLE) < 15, `skin at ${a}`);
});

test("scopeOf counts each pixel at its colour and skips the transparent", () => {
  const data = new Uint8ClampedArray([
    128, 128, 128, 255,
    255, 0, 0, 255,
    0, 0, 255, 0,
  ]);
  const size = 129;
  const { counts, peak } = scopeOf(data, size);
  const lit = [...counts.keys()].filter((k) => counts[k] > 0);
  assert.equal(lit.length, 2);
  const centre = 64 * size + 64;
  assert.equal(counts[centre], 1);
  assert.equal(peak, 1);
  assert.equal(counts.reduce((a, b) => a + b, 0), 2);
});

test("at gain 2 a colour moves twice as far and a saturated red is dropped", () => {
  const size = 129;
  const [cb, cr] = chromaOf(160, 130, 110);
  const one = cellOf(cb, cr, size, 1);
  const two = cellOf(cb, cr, size, 2);
  const dist = ([x, y]) => Math.hypot(x - 64, y - 64);
  assert.ok(dist(one) > 3);
  assert.ok(Math.abs(dist(two) - 2 * dist(one)) <= 1.5, `${dist(one)} then ${dist(two)}`);

  const red = new Uint8ClampedArray([255, 0, 0, 255]);
  assert.equal(scopeOf(red, size, 1).peak, 1);
  const doubled = scopeOf(red, size, 2);
  assert.equal(doubled.peak, 0);
  assert.equal(doubled.counts.reduce((a, b) => a + b, 0), 0);

  const mild = new Uint8ClampedArray([160, 130, 110, 255]);
  const s = scopeOf(mild, size, 2);
  assert.equal(s.counts[two[1] * size + two[0]], 1);
});
