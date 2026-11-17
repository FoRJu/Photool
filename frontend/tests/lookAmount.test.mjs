import { test } from "node:test";
import assert from "node:assert/strict";
import { scaleLook, lookIsLive, scaleMasks, lookMasksLive } from "../src/lookAmount.js";

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

test("a glow scales with the amount, its size, threshold and colour do not", () => {
  const values = { halation: 40, glow_size: 30, glow_threshold: -20, halation_colour: 25 };
  assert.equal(scaleLook(values, 50).halation, 20);
  assert.equal(scaleLook(values, 150).halation, 60);
  for (const amount of [0, 50, 150]) {
    const out = scaleLook(values, amount);
    for (const key of ["glow_size", "glow_threshold", "halation_colour"]) {
      assert.equal(out[key], values[key], `${key} at ${amount}`);
    }
  }
});

test("a look's masks scale with it, their place kept", () => {
  const sky = { kind: "sky", look: "Sky drama", exposure: -0.5, highlights: -30, saturation: 15,
    range: { kind: "luminance", lo: 40, hi: 100, smooth: 30 } };
  const depth = { kind: "depth", near: 40, far: 0, soft: 30, invert: true, exposure: 0.25 };
  const people = { kind: "people", areas: ["face_skin", "body_skin"], smooth: 25, even: 20 };
  assert.deepEqual(scaleMasks([sky], 100), [sky]);
  const [half] = scaleMasks([sky], 50);
  assert.deepEqual(half, { ...sky, exposure: -0.25, highlights: -15, saturation: 8 });
  const [d] = scaleMasks([depth], 50);
  assert.equal(d.exposure, 0.15);           // 0.125, in 0.05 steps
  assert.deepEqual([d.near, d.far, d.soft, d.invert], [40, 0, 30, true]);
  const [p] = scaleMasks([people], 150);
  assert.deepEqual(p.areas, ["face_skin", "body_skin"]);
  assert.deepEqual([p.smooth, p.even], [38, 30]);
});

test("a mask's controls are clamped to their own ranges", () => {
  const [m] = scaleMasks([{ kind: "subject", exposure: 3.5, saturation: 180, blur: 90, contrast: 90 }], 150);
  assert.deepEqual([m.exposure, m.saturation, m.blur, m.contrast], [4, 200, 100, 100]);
  const [zero] = scaleMasks([{ kind: "subject", invert: true, blur: 35, exposure: -0.2 }], 0);
  assert.deepEqual(zero, { kind: "subject", invert: true });
});

test("a look's masks are live while they read what it gives", () => {
  const look = [{ kind: "sky", look: "L", exposure: -0.5 }];
  const own = { kind: "linear", exposure: 1 };
  assert.equal(lookMasksLive([own, ...scaleMasks(look, 60)], "L", look, 60), true);
  assert.equal(lookMasksLive([own, ...scaleMasks(look, 60)], "L", look, 100), false);
  assert.equal(lookMasksLive([own], "L", look, 100), false);
  assert.equal(lookMasksLive([own, { ...look[0], exposure: -1 }], "L", look, 100), false);
});

test("the newer mask controls scale with a look, and a mask's own Amount is kept", () => {
  const m = { kind: "sky", temperature: 180, tint: -40, whites: 20, blacks: -10, hue: 120, sharpness: 30, amount: 60 };
  const [half] = scaleMasks([m], 50);
  assert.deepEqual(half, { kind: "sky", temperature: 90, tint: -20, whites: 10, blacks: -5, hue: 60,
    sharpness: 15, amount: 60 });
  const [more] = scaleMasks([m], 150);
  assert.deepEqual([more.temperature, more.hue], [200, 180]);
  assert.equal(lookMasksLive([{ ...m, look: "L", amount: 80 }], "L", [m], 100), false);
  assert.equal(lookMasksLive([{ ...m, look: "L" }], "L", [m], 100), true);
});

test("colour grading scales its strengths, its hues, balance and the vignette's shape are kept", () => {
  const values = { midtone_hue: 30, midtone_sat: 40, tone_balance: -30, shadow_sat: 20,
    vignette: -40, vignette_size: -50, vignette_feather: 30 };
  const out = scaleLook(values, 50);
  assert.deepEqual(out, { midtone_hue: 30, midtone_sat: 20, tone_balance: -30, shadow_sat: 10,
    vignette: -20, vignette_size: -50, vignette_feather: 30 });
});
