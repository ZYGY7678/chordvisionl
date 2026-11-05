/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class", '[data-theme="dark"]'],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Every colour is a CSS variable so the light and dark themes are
        // one token swap rather than two parallel sets of classes.
        base: "rgb(var(--base) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        raised: "rgb(var(--raised) / <alpha-value>)",
        line: "rgb(var(--line) / <alpha-value>)",
        ink: "rgb(var(--ink) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        faint: "rgb(var(--faint) / <alpha-value>)",
        glow: "rgb(var(--glow) / <alpha-value>)",
        glowSoft: "rgb(var(--glow-soft) / <alpha-value>)",
        signal: "rgb(var(--signal) / <alpha-value>)",
        warn: "rgb(var(--warn) / <alpha-value>)",
        danger: "rgb(var(--danger) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["Heebo", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        chord: ["'Azeret Mono'", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: { xl2: "1.25rem", xl3: "1.75rem" },
      boxShadow: {
        lift: "0 1px 2px rgb(0 0 0 / 0.14), 0 8px 24px -12px rgb(0 0 0 / 0.5)",
        deep: "0 24px 64px -24px rgb(0 0 0 / 0.65)",
        glowRing: "0 0 0 1px rgb(var(--glow) / 0.35), 0 0 32px -6px rgb(var(--glow) / 0.45)",
      },
      transitionTimingFunction: { out3: "cubic-bezier(0.22, 1, 0.36, 1)" },
      keyframes: {
        meter: {
          "0%,100%": { transform: "scaleY(0.25)" },
          "50%": { transform: "scaleY(1)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: { shimmer: "shimmer 1.6s infinite" },
    },
  },
  plugins: [],
};
