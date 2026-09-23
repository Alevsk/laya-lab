import { defineConfig } from 'vite';

export default defineConfig({
  server: { host: '127.0.0.1', port: 5173, strictPort: true },
  build: {
    target: 'es2022',
    sourcemap: true,
    // three.js alone is ~500 kB minified; it is split into its own cacheable chunk below
    chunkSizeWarningLimit: 600,
    rollupOptions: { output: { manualChunks: { three: ['three'] } } },
  },
});
