import type { ComponentChildren } from "preact";

export function Stat({ label, value, note }: { label: string; value: ComponentChildren; note?: ComponentChildren }) {
  return (
    <div class="stat">
      <span class="stat-label">{label}</span>
      <span class="stat-value">{value}</span>
      {note && <span class="stat-note">{note}</span>}
    </div>
  );
}

type SliderProps = { label: string; value: number; min: number; max: number; step?: number; shown?: string; onInput: (value: number) => void };

export function Slider({ label, value, min, max, step = 1, shown, onInput }: SliderProps) {
  return (
    <label class="control">
      <span class="control-label">{label} <output>{shown ?? value}</output></span>
      <input type="range" min={min} max={max} step={step} value={value} onInput={(e) => onInput(Number(e.currentTarget.value))} />
    </label>
  );
}

type SegmentedProps<T extends string | number> = { label: string; value: T; options: { value: T; text: string }[]; onChange: (value: T) => void };

export function Segmented<T extends string | number>({ label, value, options, onChange }: SegmentedProps<T>) {
  return (
    <div class="control">
      <span class="control-label">{label}</span>
      <div class="segmented" role="group" aria-label={label}>
        {options.map((option) => (
          <button type="button" class={option.value === value ? "on" : ""} aria-pressed={option.value === value} onClick={() => onChange(option.value)}>
            {option.text}
          </button>
        ))}
      </div>
    </div>
  );
}

export const int = (value: number) => Math.round(value).toLocaleString("en-US");
export const fixed = (value: number, digits = 2) => value.toFixed(digits).replace("-", "−");
