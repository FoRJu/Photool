import { Memories } from "./FolderOverview.jsx";
import { Rediscover, Collections } from "./Explore.jsx";

// The Explore tab: the ways in that guess or surprise. The Library tab stays a
// plain, reliable walk by date and folder; everything here is a suggestion.
export default function ExploreTab({ onPlay, onOpen, onSearch }) {
  return (
    <div className="space-y-6" data-testid="explore">
      <p className="text-xs text-neutral-500">
        Suggestions drawn from the library: what happened on this day, good photos nobody has opened lately,
        and scenes the pictures look like. To browse by date and folder, use Library.
      </p>
      <Memories onPlay={onPlay} onOpen={onOpen} />
      <Rediscover onPlay={onPlay} onOpen={onOpen} />
      <Collections onOpen={onOpen} onSearch={onSearch} />
    </div>
  );
}
