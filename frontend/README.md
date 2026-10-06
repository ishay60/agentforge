# AgentForge frontend

Vite + React 18 + TypeScript + Tailwind.

```bash
npm install
npm run dev        # http://localhost:5173, proxies /api -> http://localhost:8000
npm run build      # typecheck + production bundle in dist/
npm run typecheck
```

Env: `VITE_API_BASE` (default `""` = same origin: the Vite dev proxy or nginx forwards `/api` to the backend). Set it to `http://localhost:8000` to hit the backend directly (needs CORS).
