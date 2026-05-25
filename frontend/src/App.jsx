import { useEffect, useRef, useState } from 'react'

function Bubble({ role, content }) {
  const isUser = role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={
          `max-w-[85%] rounded-2xl border px-4 py-3 text-sm leading-relaxed ` +
          (isUser
            ? 'border-indigo-500/30 bg-indigo-500/10 text-zinc-100'
            : 'border-zinc-800/60 bg-zinc-900/40 text-zinc-100')
        }
      >
        <div className="mb-1 text-[11px] uppercase tracking-wide text-zinc-400">
          {isUser ? 'You' : 'Groq'}
        </div>
        <div className="whitespace-pre-wrap">{content}</div>
      </div>
    </div>
  )
}

function App() {
  const [draft, setDraft] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)

  const [messages, setMessages] = useState(() => [
    {
      id: crypto.randomUUID(),
      role: 'assistant',
      content:
        'Hi — I am a Groq chatbot connected to MCP tools. Ask about a support ticket (e.g. TKT-101) or product catalog questions.',
    },
  ])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end' })
  }, [messages, isSending, error])

  async function handleSend(text) {
    const trimmed = text.trim()
    if (!trimmed || isSending) return

    setError('')
    setIsSending(true)
    setDraft('')

    const nextMessages = [
      ...messages,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: trimmed,
      },
    ]
    setMessages(nextMessages)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: nextMessages.map((m) => ({ role: m.role, content: m.content })) }),
      })

      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data?.detail || data?.error || 'Request failed')
      }

      const reply = String(data?.reply || '').trim() || '(no reply)'
      setMessages((curr) => [
        ...curr,
        { id: crypto.randomUUID(), role: 'assistant', content: reply },
      ])
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="h-full bg-zinc-950 text-zinc-100">
      <div className="mx-auto flex h-full max-w-3xl flex-col">
        <header className="flex h-14 items-center justify-between border-b border-zinc-800/60 px-4">
          <div className="text-sm font-semibold tracking-tight">MCP Chatbot</div>
          <div className="text-xs text-zinc-400">Groq + MCP</div>
        </header>

        <main className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <div className="space-y-4">
            {messages.map((m) => (
              <Bubble key={m.id} role={m.role} content={m.content} />
            ))}

            {isSending ? (
              <div className="text-xs text-zinc-400">Groq is thinking…</div>
            ) : null}

            {error ? (
              <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">
                {error}
              </div>
            ) : null}
            <div ref={bottomRef} />
          </div>
        </main>

        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleSend(draft)
          }}
          className="border-t border-zinc-800/60 p-3"
        >
          <div className="flex items-center gap-2">
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Type your message…"
              className="h-10 w-full rounded-lg border border-zinc-800/60 bg-zinc-900/40 px-3 text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
              disabled={isSending}
            />
            <button
              type="submit"
              className="h-10 rounded-lg bg-indigo-600 px-4 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-60"
              disabled={isSending}
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default App
