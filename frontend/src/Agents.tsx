import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FileText, MessageSquare, Plus, Wrench } from 'lucide-react'
import { api, fmtUsd, json, type Agent, type AgentIn } from './api'

const DEFAULT_MODEL = { anthropic: 'claude-sonnet-4-5', openai: 'gpt-4o-mini' } as const

const blank = (): AgentIn => ({
  name: '', description: '',
  system_prompt: 'You are a helpful assistant. Answer from the provided documentation; if unsure, say so.',
  llm_provider: 'anthropic', llm_model: DEFAULT_MODEL.anthropic, temperature: 0.3, top_k: 5,
})

export default function Agents() {
  const qc = useQueryClient()
  const [show, setShow] = useState(false)
  const [form, setForm] = useState<AgentIn>(blank)
  const agents = useQuery({ queryKey: ['agents'], queryFn: () => api<{ agents: Agent[] }>('/agents') })
  const create = useMutation({
    mutationFn: (body: AgentIn) => api<Agent>('/agents', json(body)),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['agents'] }); setShow(false); setForm(blank()) },
  })
  const set = <K extends keyof AgentIn>(k: K, v: AgentIn[K]) => setForm((f) => ({ ...f, [k]: v }))

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Agents</h1>
        <button className="btn" onClick={() => setShow((s) => !s)}><Plus className="h-4 w-4" /> New agent</button>
      </div>

      {show && (
        <form className="card mb-6 grid gap-3 md:grid-cols-2" onSubmit={(e) => { e.preventDefault(); create.mutate(form) }}>
          <div><label className="label">Name</label><input className="input" required value={form.name} onChange={(e) => set('name', e.target.value)} /></div>
          <div><label className="label">Description</label><input className="input" value={form.description} onChange={(e) => set('description', e.target.value)} /></div>
          <div className="md:col-span-2"><label className="label">System prompt</label>
            <textarea className="input" rows={3} value={form.system_prompt} onChange={(e) => set('system_prompt', e.target.value)} /></div>
          <div><label className="label">Provider</label>
            <select className="input" value={form.llm_provider} onChange={(e) => {
              const p = e.target.value as AgentIn['llm_provider']
              setForm((f) => ({ ...f, llm_provider: p, llm_model: DEFAULT_MODEL[p] }))
            }}>
              <option value="anthropic">anthropic</option><option value="openai">openai</option>
            </select></div>
          <div><label className="label">Model</label><input className="input" value={form.llm_model} onChange={(e) => set('llm_model', e.target.value)} /></div>
          <div><label className="label">Temperature</label>
            <input className="input" type="number" min={0} max={2} step={0.1} value={form.temperature} onChange={(e) => set('temperature', Number(e.target.value))} /></div>
          <div><label className="label">Top K</label>
            <input className="input" type="number" min={1} max={20} value={form.top_k} onChange={(e) => set('top_k', Number(e.target.value))} /></div>
          <div className="flex items-center gap-3 md:col-span-2">
            <button className="btn" type="submit" disabled={create.isPending}>{create.isPending ? 'Creating…' : 'Create'}</button>
            {create.error && <span className="text-sm text-red-600">{create.error.message}</span>}
          </div>
        </form>
      )}

      {agents.isPending && <p className="text-sm text-gray-500">Loading…</p>}
      {agents.error && <p className="text-sm text-red-600">Failed to load agents: {agents.error.message}</p>}
      {agents.data?.agents.length === 0 && <p className="text-sm text-gray-500">No agents yet. Create one to get started.</p>}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {agents.data?.agents.map((a) => (
          <Link key={a.id} to={`/agents/${a.id}`} className="card hover:border-blue-400">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="truncate font-medium">{a.name}</h2>
              <span className="text-xs text-gray-500">{fmtUsd(a.total_cost_usd)}</span>
            </div>
            <div className="flex gap-4 text-xs text-gray-600">
              <span className="flex items-center gap-1"><FileText className="h-3.5 w-3.5" />{a.document_count}</span>
              <span className="flex items-center gap-1"><Wrench className="h-3.5 w-3.5" />{a.tool_count}</span>
              <span className="flex items-center gap-1"><MessageSquare className="h-3.5 w-3.5" />{a.total_conversations}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
