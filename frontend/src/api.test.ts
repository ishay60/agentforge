import { describe, expect, it } from 'vitest'
import { ApiError, sse } from './api'

const stream = (chunks: string[]) =>
  new Response(new ReadableStream({
    start(c) { for (const ch of chunks) c.enqueue(new TextEncoder().encode(ch)); c.close() },
  }), { status: 200 })

const collect = async (res: Response) => {
  const out: { event: string; data: any }[] = []
  for await (const e of sse(res)) out.push(e)
  return out
}

describe('sse', () => {
  it('parses event/data frames', async () => {
    const out = await collect(stream(['event: token\ndata: {"content":"hi"}\n\nevent: done\ndata: {"cost_usd":0}\n\n']))
    expect(out).toEqual([{ event: 'token', data: { content: 'hi' } }, { event: 'done', data: { cost_usd: 0 } }])
  })

  it('reassembles frames split across chunks, including inside multibyte characters', async () => {
    const full = 'event: token\ndata: {"content":"héllo"}\n\nevent: done\ndata: {}\n\n'
    const bytes = new TextEncoder().encode(full)
    const cut = full.indexOf('é') + 1 // split in the middle of the 2-byte é
    const res = new Response(new ReadableStream({
      start(c) { c.enqueue(bytes.slice(0, cut)); c.enqueue(bytes.slice(cut, cut + 20)); c.enqueue(bytes.slice(cut + 20)); c.close() },
    }), { status: 200 })
    const out = await collect(res)
    expect(out.map((e) => e.event)).toEqual(['token', 'done'])
    expect(out[0].data.content).toBe('héllo')
  })

  it('defaults the event name and ignores frames without data', async () => {
    const out = await collect(stream([': comment\n\ndata: {"a":1}\n\n']))
    expect(out).toEqual([{ event: 'message', data: { a: 1 } }])
  })

  it('throws ApiError on a non-2xx response', async () => {
    await expect(collect(new Response('nope', { status: 429 }))).rejects.toBeInstanceOf(ApiError)
  })
})
