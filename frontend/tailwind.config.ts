import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        forest: {
          DEFAULT: "#386641",
          hover: "#2d5234",
          light: "#497d54",
        },
        leaf: {
          DEFAULT: "#6a994e",
          hover: "#5a8442",
          light: "#7dae5f",
        },
        sage: {
          DEFAULT: "#a7c957",
          light: "#bbd876",
          faint: "#edf4dc",
        },
        parchment: {
          DEFAULT: "#f2e8cf",
          light: "#faf6ee",
          card: "#f7f1df",
          dark: "#e4d6b6",
          subtle: "#d6c5a0",
        },
        rust: {
          DEFAULT: "#bc4749",
          hover: "#a93e40",
          light: "#cf5e60",
          faint: "#faeae9",
        },
        ink: {
          DEFAULT: "#283b2a",
          muted: "#5a6e5b",
          faint: "#879b88",
        },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      boxShadow: {
        glass: "0 4px 20px -2px rgba(56, 102, 65, 0.06), 0 2px 6px -1px rgba(56, 102, 65, 0.04)",
        card: "0 10px 30px -4px rgba(56, 102, 65, 0.08), 0 4px 12px -2px rgba(56, 102, 65, 0.04)",
        elevated: "0 20px 40px -8px rgba(56, 102, 65, 0.12), 0 6px 16px -2px rgba(56, 102, 65, 0.06)",
      },
    },
  },
  plugins: [],
};
export default config;
