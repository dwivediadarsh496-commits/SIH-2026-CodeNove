import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./services/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        space: {
          950: "#050814",
          900: "#0B1228",
          800: "#111A35",
          700: "#172345",
        },
        cyan: {
          signal: "#00D9FF",
        },
        orbit: {
          purple: "#6C63FF",
          green: "#32E875",
        },
        muted: "#8491A5",
      },
      boxShadow: {
        hud: "0 0 0 1px rgba(0, 217, 255, 0.18), 0 24px 80px rgba(0, 0, 0, 0.36)",
        glow: "0 0 30px rgba(0, 217, 255, 0.2)",
      },
      backgroundImage: {
        "mission-grid":
          "linear-gradient(rgba(0,217,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(0,217,255,.08) 1px, transparent 1px)",
        "radial-space":
          "radial-gradient(circle at 72% 22%, rgba(0,217,255,.18), transparent 30%), radial-gradient(circle at 16% 12%, rgba(108,99,255,.14), transparent 28%), linear-gradient(135deg, #050814 0%, #071026 44%, #02040b 100%)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "Space Grotesk", "ui-sans-serif", "system-ui", "sans-serif"],
        telemetry: ["var(--font-mono)", "JetBrains Mono", "IBM Plex Mono", "monospace"],
        mono: ["var(--font-mono)", "JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
