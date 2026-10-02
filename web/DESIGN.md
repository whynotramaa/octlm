# octlm web design

Direction: cinematic editorial systems. The site reads like a technical publication with a data instrument
inside it. Typography and composition carry the page. Graphics show measured data, not decoration.

## Tokens

All tokens live in `src/styles/global.css` and have light and dark values.

- Surfaces: warm paper (`--bg`, `--surface`, `--surface-2`, `--surface-3`) and warm ink (`--ink`, `--ink-2`,
  `--ink-3`). Hairlines use `--line` and `--line-2`.
- Signal: `--signal`, one vermilion accent. Use it for the few points that matter, such as a passing run, a
  reverted idea, the active day or the current status. Everything else stays tonal.
- Dots: `--dot` for inactive marks in dot matrices. `.dotfield` paints a 14px dot grid behind a surface.
- Type: `--font-display` (Zarathustra from the device, Instrument Serif as fallback) for statements, big numbers and post titles. `--font-sans` (Plus
  Jakarta Sans) for body and interface text. `--font-mono` (JetBrains Mono) for labels, IDs, ticks and annotations.
- Radii: 7px controls, 8px chips, 14px figures, 18px panels.
- Shadows: none. No side borders. Separate things with tonal surfaces and horizontal hairlines.

## Patterns

- Section head: a mono section number and label on a 3-column rule, then a serif title and intro across the other
  9 columns. Set the second clause in `<em>`, which renders upright in `--signal`. No italics.
- Labels: mono, uppercase, 11 to 12px, `--ink-3`. Use `.label` or `.mono-label`.
- Buttons: `.btn` is solid ink. `.btn.ghost` is a hairline outline. Keep them compact.
- Status marks: small squares. Filled ink means done, filled signal means next, outlined means pending or
  switched.
- Data as structure: the hero atlas plots every eval run as a dot, the roadmap plots every experiment as a
  square, and meters are segmented bars. Build these at build time from the JSON in `src/data/`.
- Terminals stay dark in both themes.

## Restraint

The site should be easy on the eyes for long reading. Keep contrast soft: ink is warm graphite, not black, and dark
mode sits on `#131414`, not near-black. `--signal` is a muted terracotta. Display sizes cap near 84px for the home hero
and 54px for section titles. Rules use `--line` or `--line-2`, never full ink. Motion is a plain opacity fade on
reveal and a short rise on the hero title. No blur-in, no hover letter-spacing, no hover color swaps on static content.

## Avoid

Gradient fills, glass effects, glows, floating hover lifts, colored card backgrounds, and more than one accent
color in page chrome. Widgets inside posts may still use the categorical `--series-*` and `--hue-*` colors
because they encode data.
