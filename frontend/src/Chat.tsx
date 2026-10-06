import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { AlertTriangle, ChevronDown, ChevronRight, RotateCcw, Send, Wrench } from 'lucide-react'
import { API_BASE, fmtUsd, json, sse } from './api'

type Chunk = { source: string; page?: number | null; score: number; content?: string }
type Part =
  | { kind: 'retrieval'; chunks: Chunk[] }
  | { kind: 'tool_call'; name: string; args: unknown }
  | { kind: 'tool_result'; name: string; result: string; error?: boolean }
type Done = { model?: string; input_tokens: number; output_tokens: number; cost_usd: number; latency_ms: number }
type Msg = { role: 'user' | 'assistant'; content: string; parts: Part[]; done?: Done; error?: string }

export default function Chat({ agentId }: { agentId: string }) {
  const [cid, setCid] = useState<string | null>(null)
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const patchLast = (f: (m: Msg) => Msg) => setMsgs((ms) => [...ms.slice(0, -1), f(ms[ms.length - 1])])

  async function send() {
    const message = text.trim()
    if (!message || busy) return
    setText(''); setBusy(true)
    setMsgs((ms) => [...ms, { role: 'user', content: message, parts: [] }, { role: 'assistant', content: '', parts: [] }])
    try {
      const res = await fetch(`${API_BASE}/agents/${agentId}/chat`, json({ message, conversation_id: cid }))
      for await (const { event, data } of sse(res)) {
        if (event === 'metadata') setCid(data.conversation_id)
        else if (event === 'token') patchLast((m) => ({ ...m, content: m.content + data.content }))
        else if (event === 'retrieval') patchLast((m) => ({ ...m, parts: [...m.parts, { kind: 'retrieval', chunks: data.chunks }] }))
        else if (event === 'tool_call') patchLast((m) => ({ ...m, parts: [...m.parts, { kind: 'tool_call', name: data.name, args: data.args }] }))
        else if (event === 'tool_result') patchLast((m) => ({ ...m, parts: [...m.parts, { kind: 'tool_result', name: data.name, result: data.result, error: data.error }] }))
        else if (event === 'done') patchLast((m) => ({ ...m, done: data }))
      }
    } catch (e) {
      patchLast((m) => ({ ...m, error: (e as Error).message }))
    } finally { setBusy(false) }
  }

  return (
    <div className="flex h-[calc(100vh-14rem)] flex-col">
      <div className="mb-2 flex items-center justify-between text-xs text-gray-500">
        <span>{cid ? `Conversation ${cid.slice(0, 8)}` : 'New conversation'}</span>
        <button className="btn-secondary !py-1" onClick={() => { setCid(null); setMsgs([]) }} disabled={busy}><RotateCcw className="h-3.5 w-3.5" /> New conversation</button>
      </div>
      <div className="card flex-1 space-y-4 overflow-y-auto">
        {msgs.length === 0 && <p className="text-sm text-gray-400">Ask the agent something.</p>}
        {msgs.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-lg px-3 py-2 text-sm ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}>
              {m.parts.map((p, j) => <PartView key={j} p={p} />)}
              {m.role === 'user' ? m.content : (
                <div className="prose-chat"><ReactMarkdown>{m.content || (busy && i === msgs.length - 1 && !m.error ? '…' : '')}</ReactMarkdown></div>
              )}
              {m.error && <p className="mt-1 text-xs text-red-600">Error: {m.error}</p>}
              {m.done && (
                <p className="mt-2 text-[11px] text-gray-500">
                  {m.done.model ?? 'model?'} · {m.done.input_tokens}→{m.done.output_tokens} tok · {fmtUsd(m.done.cost_usd)} · {m.done.latency_ms} ms
                </p>
              )}
            </div>
          </div>
        ))}
        <div ref={bottom} />
      </div>
      <form className="mt-2 flex gap-2" onSubmit={(e) => { e.preventDefault(); send() }}>
        <input className="input" placeholder="Message…" value={text} onChange={(e) => setText(e.target.value)} disabled={busy} />
        <button className="btn" type="submit" disabled={busy || !text.trim()}><Send className="h-4 w-4" /></button>
      </form>
    </div>
  )
}

function PartView({ p }: { p: Part }) {
  const [open, setOpen] = useState(false)
  if (p.kind === 'retrieval') {
    return (
      <div className="mb-2 rounded border border-gray-200 bg-white text-xs">
        <button type="button" className="flex w-full items-center gap-1 px-2 py-1 font-medium text-gray-700" onClick={() => setOpen((o) => !o)}>
          {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />} Sources ({p.chunks.length})
        </button>
        {open && (
          <ul className="border-t border-gray-200 px-2 py-1">
            {p.chunks.map((c, i) => (
              <li key={i} className="py-1">
                <span className="font-medium">{c.source}</span>{c.page != null && <span> · p.{c.page}</span>} · score {c.score.toFixed(3)}
                {c.content && <p className="text-gray-500">{c.content}</p>}
              </li>
            ))}
          </ul>
        )}
      </div>
    )
  }
  if (p.kind === 'tool_call') {
    return (
      <div className="mb-2 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-xs">
        <span className="flex items-center gap-1 font-medium"><Wrench className="h-3 w-3" /> {p.name}</span>
        {p.args != null && <pre className="mt-1 whitespace-pre-wrap text-[11px] text-gray-600">{JSON.stringify(p.args)}</pre>}
      </div>
    )
  }
  return (
    <div className={`mb-2 rounded border px-2 py-1 text-xs ${p.error ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}`}>
      <span className="flex items-center gap-1 font-medium">{p.error ? <AlertTriangle className="h-3 w-3 text-red-600" /> : <Wrench className="h-3 w-3" />} {p.name} result</span>
      <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap text-[11px] text-gray-600">{p.result}</pre>
    </div>
  )
}
