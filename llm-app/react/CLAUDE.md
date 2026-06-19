# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Vite + React + TypeScript + Tailwind CSS SPA for "공공조달 어시스턴트" (public procurement assistant) agents. This used to be a Streamlit app (see git history) — it was rewritten in-place as a real React frontend, with a mobile-responsive design.

This app is the frontend only. All LLM/agent logic, conversation persistence, and MCP tool orchestration live in the separate `../../llm-server` FastAPI backend, reached via `VITE_API_BASE_URL` (see `.env` / `.env.example`, default `http://localhost:9001/api/v1`). This app never calls an LLM directly — it only calls the backend's REST endpoints under `/api/v1`: `/agents/`, `/convrstnHistory/`, `/convrstn/question`, `/upload/`.

## Commands

```cmd
npm install
npm run dev        # Vite dev server, http://localhost:5173
npm run build      # tsc -b && vite build (typecheck + production build)
npm run lint       # eslint .
npm run preview    # serve the production build locally
```

Env config: copy `.env.example` to `.env` and set `VITE_API_BASE_URL`. Vite only exposes env vars prefixed `VITE_` to client code (`import.meta.env.VITE_API_BASE_URL`, typed in `src/vite-env.d.ts`).

## Backend coupling — read before touching uploads or CORS

Browsers can't write to the backend's filesystem the way the old Streamlit app (which ran server-side) could. Two things were added to `../../llm-server` specifically to support this SPA:
- `routers/upload.py` — `POST /api/v1/upload/` (multipart), saves to `./uploadFile/<YYYYMMDD>/<uuid>_<filename>` via `utils/fileUtils.py:save_uploaded_file`, returns `{ fileFullPath, fileName }`. The frontend always uploads first, then sends the returned `fileFullPath` string in the `/convrstn/question` request body — the backend's question/notice-scan logic still works purely off that path string, unchanged.
- `main.py` — `CORSMiddleware` allowing `http://localhost:5173`/`5174`-style dev origins. If you deploy the built SPA somewhere other than localhost, add that origin here or the browser will silently block every request.

## Architecture

**Routing replaces Streamlit's session-state branching.** `src/App.tsx` wraps everything in `AppStateProvider` + `react-router-dom`:
- `/` → `pages/AgentPickerPage.tsx` — agent grid from `api/agents.ts:fetchAgentList()`. Clicking "시작하기" calls `startNewConversation(agent)` (new `convrstn_id`, empty messages) and navigates to `/agent/:mode`.
- `/agent/:mode` → `pages/AgentRouterPage.tsx`. If `selectedAgent` in context doesn't match the URL's `:mode` (e.g. page refresh or direct link), it re-fetches the agent list and resolves it via `startNewConversation` before rendering. Once resolved, it dispatches to `PpsAssistAgentPage` or `NoticeScanAgentPage` by literal `mode` string match.

Adding a new agent mode: add a branch in `AgentRouterPage.tsx`, add a new page under `src/pages/`, and make sure the backend's `/agents/` response uses a matching `mode` string — same contract as before.

**App state** (`src/context/appStateStore.ts` + `AppStateContext.tsx`) replaces Streamlit's `st.session_state`: `selectedAgent`, `convrstnId`, `messages`, `enableExtDocse`, `fileFullPath`, `sidebarOpen`, and a `historyVersion` counter used to force the sidebar's conversation list to refetch (bumped via `refreshHistory()` after a chat answer completes). Access via the `useAppState()` hook — never import the raw context object outside `appStateStore.ts`/`AppStateContext.tsx` (kept split into two files solely to satisfy the react-refresh "only export components" lint rule).

**PpsAssistAgent** (`pages/PpsAssistAgentPage.tsx`) streams: `api/qna.ts:askQuestionStreaming()` reads the backend's `text/plain` streaming response via `response.body.getReader()` and calls back per chunk, which updates the last message's `answer` in context — this is the direct equivalent of the old `stream_placeholder.markdown(full_response + "▌")` typing effect.

**NoticeScanAgent** (`pages/NoticeScanAgentPage.tsx`) is request/response, not streaming: upload → `api/qna.ts:analyzeNotice()` (same `/convrstn/question` endpoint, but the backend replies with one JSON blob) → render `extracted_data.general/execution/items` through `components/notice/SummaryMetrics.tsx` + `ResultTabs.tsx`, with a raw-JSON fallback when `extracted_data` is empty — same structure as the old Streamlit tabs (상세정보/품목/JSON/원문).

**Conversation history sidebar** (`components/sidebar/Sidebar.tsx` → `ConversationList.tsx` → `ConversationItem.tsx`, via `hooks/useConversationHistory.ts`) fetches/deletes entirely through the backend (`/convrstnHistory/`) — there's still no local DB/cache. "이어하기" (resume, in `ConversationList.tsx:handleResume`) fetches the full detail list, maps it into `ChatMessage[]`, and calls `resumeConversation()` + navigates to `/agent/:mode`.

**Responsive layout** (`components/layout/Shell.tsx`): sidebar is a static left column on `lg:` and up, and an off-canvas drawer (translate-x transition + backdrop) controlled by `sidebarOpen` on mobile, toggled from the hamburger in `Topbar.tsx`. If you add a new full-page view, mount it under the `<Route element={<Shell />}>` parent route so it inherits this shell instead of rebuilding layout chrome.

Styling is Tailwind v4 (`@tailwindcss/vite` plugin, no separate `tailwind.config.js` — theme tokens like `--color-brand-*` live in `@theme` block in `src/index.css`). Markdown answers render via `react-markdown` + `remark-gfm` inside a `prose` wrapper (`@tailwindcss/typography`).
