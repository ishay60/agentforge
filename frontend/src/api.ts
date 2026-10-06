export const API_BASE = `${import.meta.env.VITE_API_BASE ?? ''}/api/v1`

export type Agent = {
  id: string; name: string; status: string; document_count: number; tool_count: number
  total_conversations: number; total_cost_usd: number; created_at: string
}
export type AgentIn = {
  name: string; description: string; system_prompt: string; llm_provider: 'anthropic' | 'openai'
  llm_model: string; temperature: number; top_k: number
}
export type Doc = {
  id: string; filename: string; source_type: string; status: string; chunk_count: number
  file_size_bytes: number; processed_at: string
}
export type Tool = { id: string; name: string; tool_type: string; description: string; is_enabled: boolean }
export type Conversation = {
  id: string; title: string; message_count: number; total_cost_usd: number; created_at: string; updated_at: string
}
export type Message = { id: string; role: 'user' | 'assistant' | 'tool' | string; content: string; tool_calls?: unknown[] }
export type Analytics = {
  period: string
  summary: { total_conversations: number; total_messages: number; total_cost_usd: number; avg_latency_ms: number; avg_cost_per_conversation: number }
  daily: { date: string; conversations: number; messages: number; cost_usd: number }[]
  model_breakdown: { model: string; calls: number; cost_usd: number }[]
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message) }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    let detail = res.statusText
    try { detail = (await res.json()).detail ?? detail } catch { /* not json */ }
    throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json()
}

export const json = (body: unknown, method = 'POST'): RequestInit =>
  ({ method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })

/** Consume an SSE stream from a POST response. Yields {event, data}. */
export async function* sse(res: Response): AsyncGenerator<{ event: string; data: any }> {
  if (!res.ok) throw new ApiError(res.status, await res.text())
  const reader = res.body!.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i: number
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, i)
      buf = buf.slice(i + 2)
      let event = 'message'
      const dataLines: string[] = []
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
      }
      if (dataLines.length) yield { event, data: JSON.parse(dataLines.join('\n')) }
    }
  }
}

export const fmtUsd = (n: number) => `$${n.toFixed(n < 0.01 ? 5 : 3)}`
