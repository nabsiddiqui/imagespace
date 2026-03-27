import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Relative paths so output/ works at any URL — no configuration needed.
  base: './',
  // publicDir defaults to 'public/' — contains the data/ subfolder with
  // pipeline-generated files (atlases, data.bin, etc.) that Vite copies into output/.
  build: {
    // output/ is the folder users upload to GitHub Pages (or any static host).
    outDir: 'output',
  },
})
