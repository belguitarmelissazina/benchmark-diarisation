/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: "#0f1117", 2: "#161b22", 3: "#1c2128" },
        fg: { DEFAULT: "#e1e4e8", 2: "#8b949e" },
        accent: "#58a6ff",
        spk: {
          0: "#58a6ff", 1: "#3fb950", 2: "#f85149", 3: "#d29922", 4: "#bc8cff",
          5: "#f778ba", 6: "#79c0ff", 7: "#7ee787", 8: "#ffa657", 9: "#ff7b72",
        },
      },
    },
  },
  plugins: [],
};
