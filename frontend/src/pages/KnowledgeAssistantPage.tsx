import { useState, useRef, useEffect } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  grounding?: { score: number; risk: string }
  confidence?: { overall: number; tier: string }
  latency?: number
}

interface Citation {
  index: number
  title: string
  snippet: string
  score: number
}

const API = '/api'

const riskColor = (risk: string) => {
  if (risk === 'low')    return 'text-green-600 bg-green-50'
  if (risk === 'medium') return 'text-yellow-600 bg-yellow-50'
  return 'text-red-600 bg-red-50'
}

const tierColor = (tier: string) => {
  if (tier === 'high')   return 'text-green-700 bg-green-100'
  if (tier === 'medium') return 'text-yellow-700 bg-yellow-100'
  return 'text-red-700 bg-red-100'
}

export default function KnowledgeAssistantPage() {
  const [messages, setMessages]       = useState<Message[]>([])
  const [input, setInput]             = useState('')
  const [sessionId, setSessionId]     = useState<string | null>(null)
  const [template, setTemplate]       = useState('qa')
  const [multiStep, setMultiStep]     = useState(false)
  const [topK, setTopK]               = useState(5)
  const [loading, setLoading]         = useState(false)
  const [streaming, setStreaming]     = useState(false)
  const [streamBuffer, setStreamBuffer] = useState('')
  const [error, setError]             = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamBuffer])

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const userMsg = input.trim()
    setInput('')
    setError('')
    setMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setLoading(true)

    if (streaming) {
      await sendStreaming(userMsg)
    } else {
      await sendNormal(userMsg)
    }
    setLoading(false)
  }

  const sendNormal = async (message: string) => {
    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message, session_id: sessionId,
          top_k: topK, template, multi_step: multiStep,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setSessionId(data.session_id)
      setMessages(prev => [...prev, {
        role:       'assistant',
        content:    data.answer,
        citations:  data.citations,
        grounding:  data.grounding,
        confidence: data.confidence,
        latency:    data.latency_ms,
      }])
    } catch (e: any) {
      setError(e.message)
    }
  }

  const sendStreaming = async (message: string) => {
    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message, session_id: sessionId,
          top_k: topK, template,
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalCitations: Citation[] = []
      let finalGrounding: { score: number; risk: string } | undefined
      let finalConfidence: { overall: number; tier: string } | undefined
      setStreamBuffer('')

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value)
        const lines = text.split('\n')
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6)
          try {
            const evt = JSON.parse(raw)
            if (evt.type === 'session') {
              setSessionId(evt.session_id)
            } else if (evt.type === 'token') {
              buffer += evt.content
              setStreamBuffer(buffer)
            } else if (evt.type === 'done') {
              finalCitations  = evt.citations  || []
              finalGrounding  = { score: evt.grounding_score, risk: evt.hallucination_risk }
              finalConfidence = { overall: evt.confidence, tier: evt.tier }
            }
          } catch {}
        }
      }

      setStreamBuffer('')
      setMessages(prev => [...prev, {
        role:       'assistant',
        content:    buffer,
        citations:  finalCitations,
        grounding:  finalGrounding,
        confidence: finalConfidence,
      }])
    } catch (e: any) {
      setError(e.message)
    }
  }

  const clearSession = () => {
    setMessages([])
    setSessionId(null)
    setStreamBuffer('')
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4 gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Knowledge Assistant</h1>
          <p className="text-sm text-gray-500">
            RAG-powered assistant with citations, grounding verification, and confidence scoring
          </p>
        </div>
        <button
          onClick={clearSession}
          className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg"
        >
          New Chat
        </button>
      </div>

      {/* Settings bar */}
      <div className="flex flex-wrap gap-3 p-3 bg-gray-50 rounded-xl border text-sm">
        <div className="flex items-center gap-2">
          <label className="text-gray-600 font-medium">Template:</label>
          <select
            value={template}
            onChange={e => setTemplate(e.target.value)}
            className="border rounded px-2 py-1 text-sm bg-white"
          >
            {['qa','research','summarization','documentation','comparison','troubleshooting'].map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-gray-600 font-medium">Top-K:</label>
          <input
            type="number" min={1} max={20} value={topK}
            onChange={e => setTopK(Number(e.target.value))}
            className="border rounded px-2 py-1 w-16 text-sm bg-white"
          />
        </div>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={multiStep} onChange={e => setMultiStep(e.target.checked)} />
          <span className="text-gray-600">Multi-step</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={streaming} onChange={e => setStreaming(e.target.checked)} />
          <span className="text-gray-600">Stream</span>
        </label>
        {sessionId && (
          <span className="ml-auto text-xs text-gray-400 font-mono truncate max-w-[200px]">
            Session: {sessionId}
          </span>
        )}
      </div>

      {/* Message thread */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-64 text-gray-400">
            <div className="text-4xl mb-3">🔍</div>
            <p className="text-lg font-medium">Ask anything about your indexed documents</p>
            <p className="text-sm mt-1">Answers are grounded in your knowledge base with citations</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <MessageBubble key={idx} msg={msg} />
        ))}

        {/* Streaming token buffer */}
        {streamBuffer && (
          <div className="flex justify-start">
            <div className="max-w-[80%] bg-white border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
              <p className="text-sm text-gray-800 whitespace-pre-wrap">{streamBuffer}</p>
              <span className="inline-block w-1.5 h-4 bg-blue-500 animate-pulse ml-0.5" />
            </div>
          </div>
        )}

        {loading && !streamBuffer && (
          <div className="flex justify-start">
            <div className="bg-white border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
              <div className="flex gap-1">
                {[0,1,2].map(i => (
                  <div key={i} className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                       style={{ animationDelay: `${i * 150}ms` }} />
                ))}
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
            Error: {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
          placeholder="Ask a question about your documents…"
          className="flex-1 border rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="px-5 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-xl text-sm font-medium transition-colors"
        >
          {loading ? '…' : 'Send'}
        </button>
      </div>
    </div>
  )
}

function MessageBubble({ msg }: { msg: Message }) {
  const [showDetails, setShowDetails] = useState(false)

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-3">
          <p className="text-sm">{msg.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] space-y-2">
        {/* Answer bubble */}
        <div className="bg-white border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
          <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
            {msg.content}
          </p>
        </div>

        {/* Metadata row */}
        {(msg.grounding || msg.confidence || msg.latency) && (
          <div className="flex flex-wrap gap-2 items-center text-xs">
            {msg.grounding && (
              <span className={`px-2 py-0.5 rounded-full font-medium ${riskColor(msg.grounding.risk)}`}>
                Grounding: {(msg.grounding.score * 100).toFixed(0)}% · {msg.grounding.risk} risk
              </span>
            )}
            {msg.confidence && (
              <span className={`px-2 py-0.5 rounded-full font-medium ${tierColor(msg.confidence.tier)}`}>
                Confidence: {msg.confidence.tier} ({(msg.confidence.overall * 100).toFixed(0)}%)
              </span>
            )}
            {msg.latency && (
              <span className="text-gray-400">{msg.latency.toFixed(0)} ms</span>
            )}
            {(msg.citations?.length ?? 0) > 0 && (
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-blue-600 hover:underline"
              >
                {showDetails ? 'Hide' : 'Show'} {msg.citations!.length} source{msg.citations!.length > 1 ? 's' : ''}
              </button>
            )}
          </div>
        )}

        {/* Citations panel */}
        {showDetails && msg.citations && msg.citations.length > 0 && (
          <div className="space-y-2">
            {msg.citations.map(c => (
              <div key={c.index} className="bg-gray-50 border rounded-xl p-3 text-xs">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="bg-blue-100 text-blue-700 font-bold px-1.5 py-0.5 rounded">
                      [{c.index}]
                    </span>
                    <span className="font-medium text-gray-800">{c.title}</span>
                  </div>
                  <span className="text-gray-400 shrink-0">
                    {(c.score * 100).toFixed(0)}% match
                  </span>
                </div>
                {c.snippet && (
                  <p className="mt-1.5 text-gray-600 line-clamp-3">{c.snippet}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
