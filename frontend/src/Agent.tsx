import { useState, type DragEvent } from 'react'
import { useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Upload } from 'lucide-react'
import { ApiError, api, fmtUsd, json, type Agent, type Analytics, type Conversation, type Doc, type Message, type Tool } from './api'
import Chat from './Chat'

const TABS = ['Chat', 'Documents', 'Tools', 'Analytics', 'Conversations'] as const

export default function AgentPage() {
  const { id = '' } = useParams()
  const [tab, setTab] = useState<(typeof TABS)[number]>('Chat')
  const agents = useQuery({ queryKey: ['agents'], queryFn: () => api<{ agents: Agent[] }>('/agents') })
  const agent = agents.data?.agents.find((a) => a.id === id)
  if (agents.isPending) return <p className="text-sm text-gray-500">Loading…</p>
  if (agents.error) return <p className="text-sm text-red-600">{agents.error.message}</p>
  if (!agent) return <p className="text-sm text-red-600">Agent not found.</p>

  return (
    <div>
      <h1 className="mb-1 text-xl font-semibold">{agent.name}</h1>
      <p className="mb-4 text-xs text-gray-500">{agent.document_count} docs · {agent.tool_count} tools · {agent.total_conversations} conversations · {fmtUsd(agent.total_cost_usd)}</p>
      <div className="mb-4 flex gap-1 overflow-x-auto border-b border-gray-200">
        {TABS.map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`whitespace-nowrap border-b-2 px-3 py-2 text-sm ${tab === t ? 'border-blue-600 font-medium text-blue-700' : 'border-transparent text-gray-600 hover:text-gray-900'}`}>{t}</button>
        ))}
      </div>
      {tab === 'Chat' && <Chat agentId={id} />}
      {tab === 'Documents' && <Documents id={id} />}
      {tab === 'Tools' && <Tools id={id} />}
      {tab === 'Analytics' && <AnalyticsTab id={id} />}
      {tab === 'Conversations' && <Conversations id={id} />}
    </div>
  )
}

const errText = (e: Error | null) => e instanceof ApiError && e.status === 409 ? 'Already indexed.' : e?.message
const Status = ({ q }: { q: { isPending: boolean; error: Error | null } }) => (
  <>{q.isPending && <p className="text-sm text-gray-500">Loading…</p>}{q.error && <p className="text-sm text-red-600">{q.error.message}</p>}</>
)

// --- Documents ---------------------------------------------------------------

function Documents({ id }: { id: string }) {
  const qc = useQueryClient()
  const [url, setUrl] = useState('')
  const [drag, setDrag] = useState(false)
  const docs = useQuery({ queryKey: ['docs', id], queryFn: () => api<{ documents: Doc[] }>(`/agents/${id}/documents`) })
  const done = () => { qc.invalidateQueries({ queryKey: ['docs', id] }); qc.invalidateQueries({ queryKey: ['agents'] }) }
  const upload = useMutation({
    mutationFn: (file: File) => { const fd = new FormData(); fd.append('file', file); return api<Doc>(`/agents/${id}/documents`, { method: 'POST', body: fd }) },
    onSuccess: done,
  })
  const fetchUrl = useMutation({ mutationFn: (u: string) => api<Doc>(`/agents/${id}/documents`, json({ url: u })), onSuccess: () => { done(); setUrl('') } })
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) upload.mutate(f) }

  return (
    <div className="space-y-4">
      <label onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-sm ${drag ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}>
        <Upload className="mb-2 h-5 w-5 text-gray-400" />
        {upload.isPending ? 'Uploading and indexing…' : 'Drop a PDF / TXT / MD / HTML here, or click to choose'}
        <input type="file" className="hidden" accept=".pdf,.txt,.md,.html" onChange={(e) => { const f = e.target.files?.[0]; if (f) upload.mutate(f); e.target.value = '' }} />
      </label>
      {upload.error && <p className="text-sm text-red-600">{errText(upload.error)}</p>}
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (url) fetchUrl.mutate(url) }}>
        <input className="input" type="url" placeholder="https://docs.example.com/guide" value={url} onChange={(e) => setUrl(e.target.value)} />
        <button className="btn" type="submit" disabled={fetchUrl.isPending || !url}>{fetchUrl.isPending ? 'Fetching…' : 'Fetch'}</button>
      </form>
      {fetchUrl.error && <p className="text-sm text-red-600">{errText(fetchUrl.error)}</p>}
      <Status q={docs} />
      {docs.data?.documents.length === 0 && <p className="text-sm text-gray-500">No documents yet.</p>}
      {!!docs.data?.documents.length && (
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-gray-500"><tr><th className="py-1">File</th><th>Type</th><th>Status</th><th>Chunks</th><th>Size</th></tr></thead>
          <tbody>{docs.data.documents.map((d) => (
            <tr key={d.id} className="border-t border-gray-100">
              <td className="max-w-xs truncate py-2" title={d.filename}>{d.filename}</td><td>{d.source_type}</td>
              <td><span className={`rounded px-1.5 py-0.5 text-xs ${d.status === 'ready' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>{d.status}</span></td>
              <td>{d.chunk_count}</td><td>{d.file_size_bytes ? `${(d.file_size_bytes / 1024).toFixed(1)} KB` : '—'}</td>
            </tr>))}</tbody>
        </table>
      )}
    </div>
  )
}

// --- Tools ---------------------------------------------------------------------

function Tools({ id }: { id: string }) {
  const qc = useQueryClient()
  const tools = useQuery({ queryKey: ['tools', id], queryFn: () => api<{ tools: Tool[] }>(`/agents/${id}/tools`) })
  const done = () => { qc.invalidateQueries({ queryKey: ['tools', id] }); qc.invalidateQueries({ queryKey: ['agents'] }) }
  const [http, setHttp] = useState({ name: '', description: '', http_url: '', http_method: 'GET', parameters_schema: '{\n  "type": "object",\n  "properties": {}\n}' })
  const [schemaErr, setSchemaErr] = useState<string | null>(null)
  const [target, setTarget] = useState('')
  const addHttp = useMutation({ mutationFn: (body: object) => api(`/agents/${id}/tools`, json(body)), onSuccess: () => { done(); setHttp((h) => ({ ...h, name: '', description: '', http_url: '' })) } })
  const addMcp = useMutation({
    mutationFn: (t: string) => api<{ mcp_server_id: string; tools_discovered: { name: string; description: string }[] }>(`/agents/${id}/tools/mcp`, json({ target: t })),
    onSuccess: done,
  })
  const submitHttp = () => {
    let parameters_schema: unknown
    try { parameters_schema = JSON.parse(http.parameters_schema); setSchemaErr(null) } catch (e) { setSchemaErr((e as Error).message); return }
    addHttp.mutate({ ...http, parameters_schema })
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="space-y-4">
        <h2 className="font-medium">Registered tools</h2>
        <Status q={tools} />
        {tools.data?.tools.length === 0 && <p className="text-sm text-gray-500">No tools yet (built-ins are always available).</p>}
        {tools.data?.tools.map((t) => (
          <div key={t.id} className="card !p-3 text-sm">
            <div className="flex items-center gap-2"><span className="font-medium">{t.name}</span>
              <span className="rounded bg-gray-100 px-1.5 text-xs">{t.tool_type}</span>
              {!t.is_enabled && <span className="text-xs text-gray-400">disabled</span>}</div>
            <p className="text-xs text-gray-500">{t.description}</p>
          </div>
        ))}
      </div>
      <div className="space-y-6">
        <form className="card space-y-2" onSubmit={(e) => { e.preventDefault(); submitHttp() }}>
          <h2 className="font-medium">Register HTTP tool</h2>
          <input className="input" placeholder="name (a-z, 0-9, _ -)" required pattern="[a-zA-Z0-9_\-]{1,64}" value={http.name} onChange={(e) => setHttp({ ...http, name: e.target.value })} />
          <input className="input" placeholder="description" required value={http.description} onChange={(e) => setHttp({ ...http, description: e.target.value })} />
          <div className="flex gap-2">
            <select className="input !w-28" value={http.http_method} onChange={(e) => setHttp({ ...http, http_method: e.target.value })}>
              {['GET', 'POST', 'PUT', 'DELETE'].map((m) => <option key={m}>{m}</option>)}</select>
            <input className="input" type="url" placeholder="https://api.example.com/orders/{order_id}" required value={http.http_url} onChange={(e) => setHttp({ ...http, http_url: e.target.value })} />
          </div>
          <textarea className="input font-mono text-xs" rows={5} value={http.parameters_schema} onChange={(e) => setHttp({ ...http, parameters_schema: e.target.value })} />
          {schemaErr && <p className="text-xs text-red-600">Invalid JSON: {schemaErr}</p>}
          <button className="btn" type="submit" disabled={addHttp.isPending}>Register</button>
          {addHttp.error && <p className="text-xs text-red-600">{addHttp.error.message}</p>}
        </form>
        <form className="card space-y-2" onSubmit={(e) => { e.preventDefault(); if (target) addMcp.mutate(target) }}>
          <h2 className="font-medium">Connect MCP server</h2>
          <input className="input" placeholder="http://localhost:3001/mcp  or  node server.js serve" value={target} onChange={(e) => setTarget(e.target.value)} />
          <button className="btn" type="submit" disabled={addMcp.isPending || !target}>{addMcp.isPending ? 'Connecting…' : 'Connect'}</button>
          {addMcp.error && <p className="text-xs text-red-600">{addMcp.error.message}</p>}
          {addMcp.data && (
            <div className="text-xs">
              <p className="font-medium text-green-700">Discovered {addMcp.data.tools_discovered.length} tools</p>
              <ul className="list-disc pl-4">{addMcp.data.tools_discovered.map((t) => <li key={t.name}><b>{t.name}</b> — {t.description}</li>)}</ul>
            </div>
          )}
        </form>
      </div>
    </div>
  )
}

// --- Analytics --------------------------------------------------------------------

function AnalyticsTab({ id }: { id: string }) {
  const [period, setPeriod] = useState<'7d' | '30d'>('7d')
  const q = useQuery({ queryKey: ['analytics', id, period], queryFn: () => api<Analytics>(`/agents/${id}/analytics?period=${period}`) })
  const s = q.data?.summary
  const empty = q.data && q.data.summary.total_messages === 0
  return (
    <div className="space-y-4">
      <div className="flex gap-1">
        {(['7d', '30d'] as const).map((p) => (
          <button key={p} onClick={() => setPeriod(p)} className={p === period ? 'btn' : 'btn-secondary'}>{p}</button>))}
      </div>
      <Status q={q} />
      {empty && <p className="text-sm text-gray-500">No activity in this period yet.</p>}
      {s && !empty && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            {[['Conversations', s.total_conversations], ['Messages', s.total_messages], ['Cost', fmtUsd(s.total_cost_usd)],
              ['Avg latency', `${Math.round(s.avg_latency_ms)} ms`], ['Cost / conv', fmtUsd(s.avg_cost_per_conversation)]].map(([k, v]) => (
              <div key={k} className="card !p-3"><p className="text-xs text-gray-500">{k}</p><p className="text-lg font-semibold">{v}</p></div>))}
          </div>
          <div className="card">
            <h3 className="mb-2 text-sm font-medium">Daily activity</h3>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={q.data!.daily}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" /><XAxis dataKey="date" fontSize={11} /><YAxis fontSize={11} allowDecimals={false} />
                <Tooltip /><Legend />
                <Line type="monotone" dataKey="conversations" stroke="#2563eb" dot={false} />
                <Line type="monotone" dataKey="messages" stroke="#16a34a" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <h3 className="mb-2 text-sm font-medium">Cost by model</h3>
            {q.data!.model_breakdown.length === 0 ? <p className="text-sm text-gray-500">No model data.</p> : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={q.data!.model_breakdown} layout="vertical" margin={{ left: 40 }}>
                  <XAxis type="number" fontSize={11} tickFormatter={(v: number) => `$${v}`} /><YAxis type="category" dataKey="model" fontSize={11} width={140} />
                  <Tooltip formatter={(v) => fmtUsd(Number(v))} /><Bar dataKey="cost_usd" fill="#2563eb" />
                </BarChart>
              </ResponsiveContainer>)}
          </div>
        </>
      )}
    </div>
  )
}

// --- Conversations -------------------------------------------------------------------

function Conversations({ id }: { id: string }) {
  const [cid, setCid] = useState<string | null>(null)
  const convs = useQuery({ queryKey: ['convs', id], queryFn: () => api<{ conversations: Conversation[] }>(`/agents/${id}/conversations`) })
  const msgs = useQuery({ queryKey: ['msgs', id, cid], enabled: !!cid, queryFn: () => api<{ messages: Message[] }>(`/agents/${id}/conversations/${cid}/messages`) })
  return (
    <div className="grid gap-4 md:grid-cols-[16rem_1fr]">
      <div className="space-y-1">
        <Status q={convs} />
        {convs.data?.conversations.length === 0 && <p className="text-sm text-gray-500">No conversations yet.</p>}
        {convs.data?.conversations.map((c) => (
          <button key={c.id} onClick={() => setCid(c.id)} className={`block w-full rounded-md px-3 py-2 text-left text-sm ${cid === c.id ? 'bg-blue-50 text-blue-700' : 'hover:bg-gray-100'}`}>
            <p className="truncate font-medium">{c.title || c.id.slice(0, 8)}</p>
            <p className="text-xs text-gray-500">{c.message_count} msgs · {fmtUsd(c.total_cost_usd)} · {new Date(c.updated_at).toLocaleString()}</p>
          </button>))}
      </div>
      <div className="card min-h-40 space-y-3">
        {!cid && <p className="text-sm text-gray-400">Select a conversation.</p>}
        {cid && <Status q={msgs} />}
        {msgs.data?.messages.map((m) => (
          <div key={m.id} className={`text-sm ${m.role === 'user' ? 'text-right' : ''}`}>
            <span className="text-[11px] uppercase text-gray-400">{m.role}</span>
            <div className={`mt-0.5 inline-block max-w-[90%] whitespace-pre-wrap rounded-lg px-3 py-2 text-left ${m.role === 'user' ? 'bg-blue-600 text-white' : m.role === 'tool' ? 'bg-amber-50 font-mono text-xs' : 'bg-gray-100'}`}>
              {m.content || (m.tool_calls?.length ? `[called ${m.tool_calls.length} tool(s)]` : '')}
            </div>
          </div>))}
      </div>
    </div>
  )
}
