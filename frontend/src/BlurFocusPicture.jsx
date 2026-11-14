// What lens blur keeps sharp, shown rather than described (backend adjust/depth.py
// focus_picture): the frame small, in its own colours where it stays sharp and greyed
// where it is blurred, the more the more. Drawn again a moment after Focus at, Depth of
// field or the outline changes; Blur only says how strong, so it does not move it.
import { useEffect, useRef, useState } from "react";

const API = "/api";

export default function BlurFocusPicture({ body }) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const shown = useRef("");
  const key = JSON.stringify(body);

  useEffect(() => {
    const ctl = new AbortController();
    const wait = setTimeout(async () => {
      setBusy(true);
      try {
        const res = await fetch(`${API}/export/develop/focus-picture`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: key, signal: ctl.signal,
        });
        if (!res.ok) return;
        const next = URL.createObjectURL(await res.blob());
        if (shown.current) URL.revokeObjectURL(shown.current);
        shown.current = next;
        setUrl(next);
      } catch { /* the last picture stays */ } finally {
        if (!ctl.signal.aborted) setBusy(false);
      }
    }, 250);
    return () => { clearTimeout(wait); ctl.abort(); };
  }, [key]);
  useEffect(() => () => { if (shown.current) URL.revokeObjectURL(shown.current); }, []);

  return (
    <div className="space-y-1" data-testid="blur-focus-picture">
      {url ? (
        <img src={url} alt="What stays sharp: in colour; what is blurred: greyed"
          className={`w-full max-h-48 object-contain rounded border border-neutral-800 bg-neutral-950 ${busy ? "opacity-70" : ""}`} />
      ) : (
        <div className="h-24 rounded border border-neutral-800 bg-neutral-950 flex items-center justify-center text-[10px] text-neutral-600">
          Reading the distances…
        </div>
      )}
      <div className="text-[10px] text-neutral-500">In colour: sharp. Greyed: blurred, the darker the more.</div>
    </div>
  );
}
