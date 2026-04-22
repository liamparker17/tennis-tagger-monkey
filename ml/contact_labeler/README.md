# Contact Labeler

Mark the exact frame of each racket-ball contact per clip. The saved
`contact_labels.json` is consumed by `ml.point_model.dataset.ClipDataset`
and flows into `build_targets` as `strong_contact_frames`, so the
`contact_logits` head and shot-attention stroke classifier get real
supervision instead of synthetic uniform-guess contacts.

## Run

    python -m ml.contact_labeler.server --clips files/data/clips

Open http://127.0.0.1:8765

## Keys

| Key | Action |
|-----|--------|
| `,` / `.` | Step one frame back / forward |
| Space | Play / pause |
| `n` | Mark contact on the **NEAR**-court player |
| `f` | Mark contact on the **FAR**-court player |
| `u` | Undo last event |
| `s` | Save events for the current clip |
| `]` | Jump to next clip |

The header shows which player is near and which is far for the current
clip, auto-resolved from the match's preflight `setup.json` and any
`side_swaps` tagged there. Hitter index is stored canonically (0 = player_a,
1 = player_b) so labels stay correct across side swaps.

Re-opening an already-labeled clip loads its saved events, so you can
revise without re-entering everything.

The status bar shows Dartfish's stroke count hint (bucket + number of
tagged strokes) so you know roughly how many contacts to expect per
clip.
