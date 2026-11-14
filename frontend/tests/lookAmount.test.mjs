import { test } from "node:test";
import assert from "node:assert/strict";
import { scaleLook, lookIsLive } from "../src/lookAmount.js";

test("at 100 the values come back as they went in", () => {
  const values = { contrast: 20, exposure: 0.33, mono: 1, style: "s1", style_fade: 20,
    curve: [[0, 0], [128, 160], [255, 255]], shadow_hue: 210 };
  assert.deepEqual(scaleLook(values, 100), values);
});

test("at 50 a strength is halved, at 0 it is left out", () => {
  assert.deepEqual(scaleLook({ contrast: 20, exposure: 1 }, 50), { contrast: 10, exposure: 0.5 });
  assert.deepEqual(scaleLook({ contrast: 20, exposure: 1 }, 0), {});
});

test("at 150 a strength is clamped to its limits", () => {
  assert.equal(scaleLook({ vibrance: 80 }, 150).vibrance, 100);
  assert.equal(scaleLook({ exposure: 4 }, 150, { exposure: [-5, 5] }).exposure, 5);
});

test("choices are kept at any amount", () => {
  const values = { mono: 1, style: "s1", shadow_hue: 210, grain_size: 30, wb_kelvin: 5200 };
  for (const amount of [0, 50, 150]) {
    const out = scaleLook(values, amount);
    for (const key of Object.keys(values)) assert.equal(out[key], values[key], `${key} at ${amount}`);
  }
});

test("a curve bends less, never more", () => {
  const curve = [[0, 0], [128, 160], [255, 255]];
  assert.deepEqual(scaleLook({ curve }, 50).curve[1], [128, 144]);
  assert.deepEqual(scaleLook({ curve }, 150).curve, curve);
  assert.equal("curve" in scaleLook({ curve }, 0), false);
});

test("a style fades toward the camera's own look", () => {
  assert.equal(scaleLook({ style: "s1", style_fade: 20 }, 50).style_fade, 60);
  assert.equal(scaleLook({ style: "s1", style_fade: 20 }, 0).style_fade, 100);
  assert.equal(scaleLook({ style: "s1", style_fade: 20 }, 150).style_fade, 20);
  assert.equal(scaleLook({ style_fade: 20 }, 50).style_fade, 20);
});

test("a look is live while its keys read what it gives", () => {
  const keys = ["contrast", "vibrance", "curve"];
  const zero = { contrast: 0, vibrance: 0, curve: null };
  const values = { contrast: 20 };
  assert.ok(lookIsLive({ contrast: 10, vibrance: 0, curve: null, exposure: 1 }, keys, values, 50, zero));
  assert.ok(!lookIsLive({ contrast: 12, vibrance: 0, curve: null }, keys, values, 50, zero));
  assert.ok(lookIsLive({ contrast: 0, vibrance: 0, curve: null }, keys, values, 0, zero));
});
