// Full screen, the standard way or WebKit's (Safari and Chrome on an iPad). An
// iPhone lets no page have the whole screen but a video; there, Photool added
// to the home screen opens without the address bar instead.

export function fullscreenElement() {
  return document.fullscreenElement || document.webkitFullscreenElement || null;
}

export function canFullscreen() {
  const el = document.documentElement;
  return !!(el.requestFullscreen || el.webkitRequestFullscreen);
}

export function enterFullscreen(el = document.documentElement) {
  const go = el.requestFullscreen || el.webkitRequestFullscreen;
  try { go?.call(el)?.catch?.(() => {}); } catch { /* refused: the page stays as it is */ }
}

export function exitFullscreen() {
  if (!fullscreenElement()) return;
  const go = document.exitFullscreen || document.webkitExitFullscreen;
  try { go?.call(document)?.catch?.(() => {}); } catch { /* already out */ }
}

export function toggleFullscreen(el) {
  if (fullscreenElement()) exitFullscreen();
  else enterFullscreen(el);
}
