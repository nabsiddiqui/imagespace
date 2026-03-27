/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'rp-base':    '#faf4ed',
        'rp-surface': '#fffaf3',
        'rp-overlay': '#f2e9e1',
        'rp-muted':   '#9893a5',
        'rp-subtle':  '#797593',
        'rp-text':    '#575279',
        'rp-love':    '#b4637a',
        'rp-gold':    '#ea9d34',
        'rp-rose':    '#d7827e',
        'rp-pine':    '#286983',
        'rp-foam':    '#56949f',
        'rp-iris':    '#907aa9',
        'rp-hlLow':   '#f4ede8',
        'rp-hlMed':   '#dfdad9',
        'rp-hlHigh':  '#cecacd',
      },
      boxShadow: {
        'rp':    '0 2px 12px -2px rgba(87, 82, 121, 0.12)',
        'rp-lg': '0 8px 30px -4px rgba(87, 82, 121, 0.18)',
      },
      fontFamily: {
        'sans': ['"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
