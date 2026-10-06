import { useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Bot, Menu, X } from 'lucide-react'
import { api, type Agent } from './api'
import Agents from './Agents'
import AgentPage from './Agent'

export default function App() {
  const [open, setOpen] = useState(false)
  const agents = useQuery({ queryKey: ['agents'], queryFn: () => api<{ agents: Agent[] }>('/agents') })
  const link = ({ isActive }: { isActive: boolean }) =>
    `block truncate rounded-md px-3 py-2 text-sm ${isActive ? 'bg-blue-50 text-blue-700 font-medium' : 'text-gray-700 hover:bg-gray-100'}`

  return (
    <div className="flex min-h-screen">
      <aside className={`fixed inset-y-0 left-0 z-20 w-60 transform border-r border-gray-200 bg-white p-4 transition-transform md:static md:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="mb-6 flex items-center justify-between">
          <NavLink to="/agents" className="flex items-center gap-2 text-lg font-semibold" onClick={() => setOpen(false)}>
            <Bot className="h-5 w-5 text-blue-600" /> AgentForge
          </NavLink>
          <button className="md:hidden" onClick={() => setOpen(false)} aria-label="Close menu"><X className="h-5 w-5" /></button>
        </div>
        <NavLink to="/agents" end className={link} onClick={() => setOpen(false)}>All agents</NavLink>
        <div className="mt-4 mb-1 px-3 text-xs font-semibold uppercase text-gray-400">Agents</div>
        {agents.data?.agents.map((a) => (
          <NavLink key={a.id} to={`/agents/${a.id}`} className={link} onClick={() => setOpen(false)}>{a.name}</NavLink>
        ))}
        {agents.data?.agents.length === 0 && <p className="px-3 text-xs text-gray-400">None yet</p>}
      </aside>
      {open && <div className="fixed inset-0 z-10 bg-black/30 md:hidden" onClick={() => setOpen(false)} />}
      <main className="min-w-0 flex-1">
        <div className="flex items-center gap-2 border-b border-gray-200 bg-white px-4 py-2 md:hidden">
          <button onClick={() => setOpen(true)} aria-label="Open menu"><Menu className="h-5 w-5" /></button>
          <span className="font-semibold">AgentForge</span>
        </div>
        <div className="mx-auto max-w-5xl p-4 md:p-6">
          <Routes>
            <Route path="/" element={<Navigate to="/agents" replace />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/agents/:id" element={<AgentPage />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
