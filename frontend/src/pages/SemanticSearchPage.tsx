import React, { useState } from 'react'
import { api, SemanticResultItem } from '../api/client'

export default function SemanticSearchPage() {
  const [query,   setQuery  ] = useState('')
  const [results, setResults] = useState<SemanticResultItem[]>([])
  const [meta,    setMeta   ] = useState<{model:string; ms:number; total:number} | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError  ] = useState<string | null>(null)
  const [notReady, setNotReady] = useState(false)

  const search = async () => {
    if (!query.trim()) return
    setLoading(true); setError(null); setNotReady(false)
    try {
      const r = await api.semanticSearch(query)
      setResults(r.results)
      setMeta({ model: r.model, ms: r.search_time_ms, total: r.total_results })
      if (r.total_results === 0 && r.model.startsWith('mock')) setNotReady(true)
    } catch (e) { setError(String(e)) }
    finally { setLoading(false) }
  }

  const scoreColor = (s: number) =>
    s > 0.8 ? '#22c55e' : s > 0.6 ? '#facc15' : '#94a3b8'

  return (
    <div>
      <h1>Semantic Search</h1>
      <p style={{ color: '#64748b', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
        Finds semantically similar documents even without exact keyword matches.
        Powered by <strong style={{ color: '#38bdf8' }}>FAISS</strong> + dense embeddings.
        Run <code style={{ background: '#1e293b', padding: '2px 6px', borderRadius: 4 }}>
          POST /embeddings/reindex
        </code> first to build the vector index.
      </p>

      <div className="search-bar">
        <input
          type="text" placeholder='e.g. "fast data retrieval" or "neural ranking"'
          value={query} onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && search()}
        />
        <button onClick={search}>Search</button>
      </div>

      {loading && <p className="loading">Embedding query and searching vectors…</p>}
      {error   && <p className="error-msg">{error}</p>}

      {notReady && (
        <div className="card" style={{ borderColor: '#f59e0b' }}>
          <p style={{ color: '#fbbf24' }}>
            The vector index is empty or using a mock provider.
            Index some documents first then run <strong>POST /embeddings/reindex</strong>.
          </p>
        </div>
      )}

      {meta && !notReady && (
        <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '1rem' }}>
          {meta.total} results · {meta.ms} ms · model: <code>{meta.model}</code>
        </p>
      )}

      {results.length === 0 && meta && !notReady && (
        <div className="empty">No semantically similar documents found for "{query}"</div>
      )}

      {results.map(r => (
        <div key={r.doc_id} className="result-item">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="title">#{r.rank} &nbsp; {r.title}</div>
            <div style={{
              background: scoreColor(r.semantic_score),
              color: '#0f172a', fontWeight: 700, fontSize: '0.8rem',
              padding: '2px 8px', borderRadius: 9999,
            }}>
              cos = {r.semantic_score.toFixed(3)}
            </div>
          </div>
          <div className="snippet">{r.snippet}</div>
          <div className="meta">
            <span>doc_id: {r.doc_id}</span>
            <span style={{ color: '#475569' }}>chunk: {r.chunk_id}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
