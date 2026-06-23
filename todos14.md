# todos14 — Board accessibility

Found during a manual two-player user test (invite link + matchmaking queue, 9×9 and
10×10 connection wins). The game flows worked end-to-end; this is the one accessibility
gap worth addressing.

## Board is invisible to assistive tech

The board renders as an SVG of `<polygon class="Cell …">` elements with **no
accessibility semantics**:

- Cells are not in the accessibility tree — no `role`, no `aria-label`, no coordinate
  (e.g. "e5") exposed on each cell.
- There is no keyboard path to play: cells are clickable via mouse/pointer only.

A screen-reader or keyboard-only user can read the move list, clocks, and result, but
**cannot identify cells or make a move**.

### Suggested work

- [ ] Give each cell an accessible name = its Hex coordinate (e.g. `aria-label="e5"`),
      plus state ("empty" / "Red" / "Blue").
- [ ] Make cells focusable and playable from the keyboard (roving `tabindex`, arrow-key
      movement across the hex grid, Enter/Space to place a stone).
- [ ] Expose the board as a labelled grid (`role="grid"`/`gridcell` or equivalent) so the
      structure is navigable.
- [ ] Announce turn changes and the win/loss result via an `aria-live` region (the result
      text already updates — confirm it's announced).
- [ ] Verify focus states are visible (not just hover/color).

### Notes / context

- Observed on 9×9 and 10×10 boards; cells are document-order, row-major.
- Everything else (lobby, profile, registration, move list, clocks, result banners,
  live online/playing counters) is already in the a11y tree and reads fine.
