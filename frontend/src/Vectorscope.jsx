// ── The vectorscope, drawn ──────────────────────────────────────────────
//
// A measurement of the edited picture, not a change to it: every pixel placed
// by its colour, neutral in the centre. It is handed a scope worked out from
// the preview (scopeOfBitmap) and only paints it, with the six chart colours
// and the skin line over it.

import { useEffect, useMemo, useRef } from "react";
import { chromaOf, scopeOf, SKIN_ANGLE, TARGETS } from "./vectorscope.js";

// Draw the bitmap 256 px wide, read it and count it, as histogramOf does.
export function scopeOfBitmap(bitmap, gain = 1, size = 129) {
  const width = 256;
  const height = Math.max(1, Math.round((bitmap.height / bitmap.width) * width));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  const { data } = ctx.getImageData(0, 0, width, height);
  return scopeOf(data, size, gain);
}

// The colour each cell stands for: its Cb and Cr at a mid brightness, brought
// up so the brightest channel is light enough to read on black.
function cellColours(size, gain) {
  const out = new Uint8ClampedArray(size * size * 3);
  const y = 128;
  for (let row = 0; row < size; row++) {
    const cr = (1 - (row / (size - 1)) * 2) / gain;
    for (let col = 0; col < size; col++) {
      const cb = ((col / (size - 1)) * 2 - 1) / gain;
      const r = y + cr * 127.5 * 1.5748;
      const b = y + cb * 127.5 * 1.8556;
      const g = (y - 0.2126 * r - 0.0722 * b) / 0.7152;
      const lift = 220 / Math.max(1, r, g, b);
      const k = (row * size + col) * 3;
      out[k] = r * lift;
      out[k + 1] = g * lift;
      out[k + 2] = b * lift;
    }
  }
  return out;
}

export default function Vectorscope({ scope, gain = 1, probe, className = "" }) {
  const canvasRef = useRef(null);
  const size = scope?.size || 129;
  const colours = useMemo(() => cellColours(size, gain), [size, gain]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(size, size);
    if (scope && scope.peak > 0) {
      const scale = Math.log(1 + scope.peak);
      for (let k = 0; k < size * size; k++) {
        const n = scope.counts[k];
        if (!n) continue;
        image.data[k * 4] = colours[k * 3];
        image.data[k * 4 + 1] = colours[k * 3 + 1];
        image.data[k * 4 + 2] = colours[k * 3 + 2];
        image.data[k * 4 + 3] = Math.max(40, (Math.log(1 + n) / scale) * 255);
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [scope, colours, size]);

  // Drawn in the canvas's own cells, so a mark sits on the cell its colour
  // fills: chroma (cb, cr) is at the centre of cell ((cb+1)/2 (size-1), ...).
  const half = (size - 1) / 2;
  const at = (cb, cr) => [(cb + 1) * half + 0.5, (1 - cr) * half + 0.5];
  const inside = (cb, cr) => Math.abs(cb) <= 1 && Math.abs(cr) <= 1;
  const [cx, cy] = at(0, 0);
  const stroke = { stroke: "#525252", strokeWidth: 1, vectorEffect: "non-scaling-stroke", fill: "none" };
  const skin = (SKIN_ANGLE * Math.PI) / 180;
  const [sx, sy] = at(Math.cos(skin), Math.sin(skin));
  const box = size * 0.045;
  const ring = probe ? chromaOf(...probe).map((v) => v * gain) : null;

  return (
    <div className={`relative w-full max-w-[220px] mx-auto aspect-square bg-neutral-950 rounded border border-neutral-800 overflow-hidden ${className}`}>
      <canvas ref={canvasRef} width={size} height={size} data-testid="vectorscope"
        className="absolute inset-0 w-full h-full" />
      <svg viewBox={`0 0 ${size} ${size}`} className="absolute inset-0 w-full h-full pointer-events-none">
        <circle cx={cx} cy={cy} r={half * gain} {...stroke} />
        {/* The cross leaves its middle open, so a grey's dot shows. */}
        {[[-1, 0], [1, 0], [0, -1], [0, 1]].map(([dx, dy]) => (
          <line key={`${dx}${dy}`} {...stroke}
            x1={cx + dx * box * 0.5} y1={cy + dy * box * 0.5} x2={cx + dx * box * 1.5} y2={cy + dy * box * 1.5} />
        ))}
        <line x1={cx} y1={cy} x2={sx} y2={sy} {...stroke} />
        {TARGETS.map(({ label, at: [cb, cr] }) => {
          const x = cb * gain;
          const y = cr * gain;
          if (!inside(x, y)) return null;
          const [px, py] = at(x, y);
          // The letter alone marks where the chart colour falls.
          return (
            <g key={label}>
              <text x={px} y={py} fill="#737373" fontSize={size * 0.065}
                textAnchor="middle" dominantBaseline="central">{label}</text>
            </g>
          );
        })}
        {ring && inside(ring[0], ring[1]) && (
          <circle cx={at(...ring)[0]} cy={at(...ring)[1]} r={size * 0.04}
            fill="none" stroke="#ffffff" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        )}
      </svg>
    </div>
  );
}
