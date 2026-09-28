You are designing a premium interactive ML learning platform that combines the clarity of a high-end editorial website with the polish of a modern Apple-quality product interface.

The design reference establishes the visual direction: extremely clean, spacious, precise, soft, minimal, and highly legible. Do not treat this as a conventional developer blog or an analytics dashboard. It should feel like a beautifully designed technical publication that happens to contain interactive experiments.

CORE AESTHETIC

The dominant design principle is CLARITY.

Everything should have enough breathing room to be immediately understandable. Prefer whitespace, alignment, hierarchy, and typography over decoration. The interface should feel calm and almost frictionless.

Use:
- Large amounts of whitespace
- Precise alignment
- Strong typographic hierarchy
- Generous padding
- Wide margins
- Controlled content widths
- Soft rounded containers
- Extremely subtle borders
- Very soft shadows
- Near-white surfaces
- Restrained grayscale
- Minimal visual noise
- Consistent geometry throughout the entire product

Avoid:
- Dense dashboards
- Excessive cards
- Excessive gradients
- Strong drop shadows
- Thick borders
- Highly saturated backgrounds
- Random colors
- Excessive glassmorphism
- Excessive animations
- Decorative UI that does not communicate anything
- Generic SaaS dashboard aesthetics
- "AI website" visual clichés
- Neon cyberpunk styling
- Overly futuristic decoration

The product should feel premium because of its restraint and precision, not because of visual complexity.

--------------------------------------------------
TYPOGRAPHY
--------------------------------------------------

Use Plus Jakarta Sans as the primary typeface.

Google Sans Flex is an acceptable alternative where a more Google-like product character improves the interface. Manrope may be used selectively if it produces better typographic balance.

Typography must be extremely clean and readable.

Recommended hierarchy:

Display:
- Large
- Strong
- Tight but comfortable tracking
- Semibold/Bold

Section headings:
- Semibold
- Clear hierarchy
- Moderate negative tracking

Body:
- Regular/medium
- Generous line-height
- Comfortable reading width

Labels:
- Medium/Semibold
- Compact
- Never overly bold

Avoid excessive font-weight variation.

Do not use typography as decoration. Typography exists to establish hierarchy and make complex ML concepts easier to understand.

Long-form content should generally remain within approximately 680-760px for comfortable reading.

--------------------------------------------------
LAYOUT
--------------------------------------------------

Use a strong editorial grid.

The primary content should have generous horizontal margins and a controlled maximum width.

The page should never feel stretched simply because the viewport is large.

Use large vertical spacing between conceptual sections.

Think in terms of:

Section
    ↓
Breathing room
    ↓
Explanation
    ↓
Interactive visualization
    ↓
Observation
    ↓
Next concept

rather than:

Card
Card
Card
Card
Card

The ML playground should feel like a continuous learning experience rather than a collection of dashboard widgets.

--------------------------------------------------
SURFACES
--------------------------------------------------

Default surfaces should be extremely light and neutral.

Use near-white backgrounds rather than pure white everywhere.

Containers should generally have:
- 16-24px corner radius
- 1px subtle border
- Very soft shadow where elevation is necessary
- Generous internal padding

Avoid excessive containers.

Not every piece of content needs to live inside a card.

Use containers primarily when they establish hierarchy, contain an interactive experiment, isolate a visualization, or group related controls.

--------------------------------------------------
CHIP / STATUS COMPONENT LANGUAGE
--------------------------------------------------

Colored components must follow a very specific chip language.

Do NOT apply the chip treatment to the entire interface.

Chips are semantic components only:
- Status
- State
- Category
- Metadata
- Small interactive indicators
- Progress/state labels

The chip aesthetic is based on a soft semantic color system.

Each colored chip is derived from ONE semantic hue.

Never randomly mix hues within a single chip.

The structure is:

ONE SEMANTIC HUE
        |
        +-- very light / low saturation
        |       -> surface
        |
        +-- slightly darker / slightly more saturated
        |       -> lower gradient
        |
        +-- medium saturation
        |       -> border
        |
        +-- dark / highly saturated
                -> icon + text

The hue should remain approximately constant.

Only saturation and lightness should meaningfully change between layers.

For example:

PURPLE
- Surface = very light lavender
- Bottom = slightly deeper lavender
- Border = medium lavender
- Foreground = saturated deep violet

GREEN
- Surface = very light mint
- Bottom = slightly deeper mint
- Border = medium green
- Foreground = saturated deep green

BLUE
- Surface = very light blue
- Bottom = slightly deeper blue
- Border = medium blue
- Foreground = saturated deep blue

ORANGE
- Surface = very light warm cream
- Bottom = slightly deeper amber tint
- Border = medium amber
- Foreground = saturated orange

RED
- Surface = very light pink
- Bottom = slightly deeper pink
- Border = medium red
- Foreground = saturated red

CYAN
- Surface = very light cyan
- Bottom = slightly deeper cyan
- Border = medium teal
- Foreground = saturated teal

The chip background should use a SUBTLE VERTICAL GRADIENT.

Top:
- Higher lightness
- Lower saturation

Bottom:
- Slightly lower lightness
- Slightly higher saturation

The gradient must be subtle enough that it feels like a soft physical surface rather than a colorful gradient.

Conceptually:

TOP
Light + Low Saturation
        ↓
Middle
        ↓
BOTTOM
Slightly darker + Slightly more saturated

The border should be derived from the same hue and have low opacity.

The icon and text should use the same semantic foreground color.

Recommended chip geometry:
- Height: approximately 44-52px depending on context
- Radius: approximately 12-15px
- Horizontal padding: 16-20px
- Icon: 20-24px
- Icon/text gap: 8-10px
- Font weight: 500-600

The chip should feel tactile, soft, and slightly elevated.

Use extremely soft shadows.

Do NOT make chips look like:
- Neon badges
- Strong gradients
- Pills with aggressive saturation
- Dark UI tags
- Heavy glassmorphism

They should feel like subtle premium status indicators.

--------------------------------------------------
INTERACTIVE ML PLAYGROUND
--------------------------------------------------

The core experience is an interactive playground for understanding machine learning.

The interface should communicate:

"I can understand this."

Not:

"This website has many features."

Each concept should follow a visual learning rhythm:

1. Concept introduction
2. Intuition
3. Interactive visualization
4. User interaction
5. Observation
6. Mathematical explanation
7. Practical interpretation

Example:

        SECTION LABEL

        Understanding Gradient Descent

        Short explanation of the intuition.

        ┌─────────────────────────────────────┐
        │                                     │
        │       Interactive Visualization     │
        │                                     │
        │       graph / animation / model     │
        │                                     │
        │       controls                      │
        │                                     │
        └─────────────────────────────────────┘

        What happened?

        Concise explanation.

        Mathematical intuition

        equation / diagram

The visualization should be the visual focus of the section.

Do not surround every element with a card.

--------------------------------------------------
INTERACTIVE CONTROLS
--------------------------------------------------

Controls should inherit the same design philosophy.

Use:
- Rounded controls
- Soft borders
- Generous padding
- Clear labels
- Minimal shadows
- Smooth hover/focus states

When a control has a semantic color, use the chip color system.

Controls should never become visually louder than the visualization they control.

Interactions should feel responsive but restrained.

Use subtle transitions around 150-250ms.

Avoid excessive bouncing, scaling, glowing, or spring animations.

--------------------------------------------------
COLOR SYSTEM
--------------------------------------------------

The base UI should be predominantly neutral.

Use grayscale for:
- Background
- Main text
- Secondary text
- Borders
- Navigation
- Most layout elements

Use semantic colors only when they communicate meaning.

Color should answer a question.

Examples:

Blue = information
Green = success/convergence/correct state
Orange = attention/warning
Red = error/destructive state
Purple = special/model-related state
Cyan = active/synchronized/interactive state

Do not introduce colors merely to make the page "look interesting."

ML visualizations may use additional colors when the color is mathematically or conceptually meaningful.

--------------------------------------------------
ICONS
--------------------------------------------------

Use simple outline icons.

Iconography should be:
- Minimal
- Geometric
- Consistent
- Approximately 20-24px
- Consistent stroke weight

Icons should never dominate text.

Avoid decorative icon collections.

--------------------------------------------------
BORDERS + SHADOWS
--------------------------------------------------

Borders should be extremely subtle.

Prefer:

1px neutral border

over:

2px dark border.

Shadows should communicate elevation, not decoration.

Use:
- Low opacity
- Large blur
- Small vertical offset

Never use harsh black shadows.

Floating elements should feel like they are gently sitting above the page.

--------------------------------------------------
ROUNDNESS
--------------------------------------------------

Use a consistent radius system.

Suggested:

Small controls:
8-10px

Chips:
12-15px

Cards / visualization containers:
16-24px

Large panels:
24-32px

Do not randomly change border radius between components.

The entire product should feel like it belongs to one geometric system.

--------------------------------------------------
NAVIGATION
--------------------------------------------------

Navigation should be extremely simple.

Do not create a giant SaaS navigation bar.

Use a clean header with:
- Brand
- Small number of navigation items
- Search if necessary
- Minimal utility controls

Maintain generous spacing.

The header should visually disappear into the page rather than dominate it.

--------------------------------------------------
CONTENT DESIGN
--------------------------------------------------

This is an educational product, so content is the primary interface.

Use short paragraphs.

Break complex explanations into digestible sections.

Use:
- Strong headings
- Small section labels
- Short explanations
- Mathematical notation
- Interactive diagrams
- Code examples
- Observations
- Callouts
- Small semantic chips

Avoid walls of text.

The design should make technical material feel approachable without making it feel childish.

--------------------------------------------------
CODE BLOCKS
--------------------------------------------------

Code should feel like part of the editorial system.

Use a restrained code surface with:
- Soft background
- Rounded corners
- Subtle border
- Generous padding
- Excellent syntax contrast
- Clear line height

Do not make code blocks visually louder than interactive visualizations.

--------------------------------------------------
MATHEMATICS
--------------------------------------------------

Mathematical notation should have generous whitespace.

Equations should never feel cramped.

Use typography and spacing to distinguish:

intuition
    ↓
formula
    ↓
interpretation

The mathematics should feel like an integral part of the article rather than an inserted academic document.

--------------------------------------------------
ANIMATION
--------------------------------------------------

Animation should communicate state or causality.

Good:
- Gradient descent moving toward a minimum
- Data points appearing
- Decision boundaries changing
- Parameters updating
- Smooth graph transitions
- Subtle hover states

Bad:
- Decorative particles
- Constant floating elements
- Excessive parallax
- Large entrance animations
- Excessive spring physics
- Animation for its own sake

The interface should remain calm even while the visualization is active.

--------------------------------------------------
RESPONSIVE DESIGN
--------------------------------------------------

The design must work beautifully from large desktop screens down to mobile.

On smaller screens:
- Preserve whitespace where possible
- Reduce horizontal padding intelligently
- Collapse navigation
- Stack visualization controls
- Preserve readable text widths
- Never allow interactive graphs to become unusably small

Do not simply shrink the desktop design.

Recompose it.

--------------------------------------------------
QUALITY BAR
--------------------------------------------------

Before considering a component complete, ask:

1. Is the purpose immediately obvious?
2. Is there enough whitespace?
3. Is the hierarchy obvious without color?
4. Is the typography exceptionally readable?
5. Are all elements aligned to the same visual system?
6. Is the component unnecessarily enclosed in a card?
7. Is the color communicating something meaningful?
8. Is the interaction visually quiet?
9. Does the component feel like it belongs to the same product?
10. Can anything be removed without losing information?

If something can be removed without reducing comprehension, remove it.

--------------------------------------------------
FINAL DESIGN PRINCIPLE
--------------------------------------------------

Build this as a premium interactive ML publication, not a dashboard.

The visual language should combine:

EDITORIAL CLARITY
+
PRODUCT-LEVEL POLISH
+
SOFT GEOMETRY
+
GENEROUS WHITESPACE
+
RESTRAINED COLOR
+
SEMANTIC CHIP SYSTEM
+
INTERACTIVE VISUAL LEARNING

The result should feel exceptionally clean, intelligent, calm, and intentional.

The user should never wonder where to look.

The user should never feel overwhelmed by the interface.

The interface should disappear enough that the ML concept becomes the experience.
