import React, { useState } from 'react'
import { api, HybridResultItem } from '../api/client'

type Mode = 'hybrid' | 'bm25' | 'semantic'

export default function HybridSearchPage() {
  const [query,   setQuery   ] = useState('')
  const [mode,    setMode    ] = useState<Mode>('hybrid')
  const [results, setResults ] = useState<HybridResultItem[]>([])
  const [meta,    setMeta    ] = useState<Record<string,unknown> | null>(null)
  const [loading, setLoading ] = useState(false)
  const [error,   setError   ] = useState<string | null>(null)

  const search = async () => {
    if (!query.trim()) return
    setLoading(true); setError(null)
    try {
      if (mode === 'hybrid') {
        const r = await api.hybridSearch(query)
        setResults(r.results)
        setMeta({
          fusion: r.fusion_strategy,
          ms: r.search_time_ms,
          bm25_count: r.bm25_results,
          sem_count:  r.semantic_results,
          total: r.total_results,
        })
      } else if (mode === 'semantic') {
        const r = await api.semanticSearch(query)
        setResults(r.results.map((x, i) => ({
          rank: x.rank, doc_id: x.doc_id, title: x.title,
          snippet: x.snippet, fusion_score: x.semantic_score,
          bm25_score: 0, semantic_score: x.semantic_score,
          bm25_rank: null, semantic_rank: i + 1,
        })))
        setMeta({ ms: r.search_time_ms, total: r.total_results })
      } else {
        const r = await api.search(query)
        setResults(r.results.map((x, i) => ({
          rank: x.rank, doc_id: x.doc_id, title: x.title,
          snippet: x.snippet, fusion_score: x.score,
          bm25_score: x.score, semantic_score: 0,
          bm25_rank: i + 1, semantic_rank: null,
        })))
        setMeta({ ms: r.search_time_ms, total: r.total_matches })
      }
    } catch (e) { setError(String(e)) }
    finally { setLoading(false) }
  }

  return (
    <div>
      <h1>Hybrid Search</h1>
      <p style={{ color: '#64748b', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
        Combines BM25 keyword matching and semantic vector search via
        <strong style={{ color: '#38bdf8' }}> Reciprocal Rank Fusion (RRF)</strong>.
        Compare modes side-by-side to understand where each method excels.
      </p>

      {/* Mode selector */}
      <div className="tabs" style={{ marginBottom: '1rem' }}>
        {(['hybrid', 'bm25', 'semantic'] as Mode[]).map(m => (
          <button key={m} className={`tab ${mode === m ? 'active' : ''}`}
            onClick={() => setMode(m)}>
            {m === 'hybrid' ? 'Hybrid (RRF)' : m === 'bm25' ? 'BM25 Only' : 'Semantic Only'}
          </button>
        ))}
      </div>

      <div className="search-bar">
        <input type="text" placeholder='Search with hybrid intelligence…'
          value={query} onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && search()} />
        <button onClick={search}>Search</button>
      </div>

      {loading && <p className="loading">Searching…</p>}
      {error   && <p className="error-msg">{error}</p>}

      {meta && (
        <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '1rem' }}>
          {String(meta.total)} results · {String(meta.ms)} ms
          {meta.fusion && <> · fusion: <strong>{String(meta.fusion)}</strong></>}
          {meta.bm25_count !== undefined &&
            <> · BM25: {String(meta.bm25_count)} / Semantic: {String(meta.sem_count)}</>}
        </p>
      )}

      {results.length === 0 && meta && (
        <div className="empty">No results for "{query}" using {mode} mode</div>
      )}

      {results.map(r => (
        <div key={r.doc_id} className="result-item">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div className="title">#{r.rank} &nbsp; {r.title}</div>
            <div style={{ display: 'flex', gap: '0.5rem', flexShrink: 0 }}>
              {r.bm25_score > 0 && (
                <span className="badge badge-blue">BM25 {r.bm25_score.toFixed(3)}</span>
              )}
              {r.semantic_score > 0 && (
                <span className="badge badge-green">cos {r.semantic_score.toFixed(3)}</span>
              )}
              {mode === 'hybrid' && (
                <span className="badge badge-yellow">RRF {r.fusion_score.toFixed(4)}</span>
              )}
            </div>
          </div>
          <div className="snippet">{r.snippet}</div>
          <div className="meta">
            <span>doc_id: {r.doc_id}</span>
            {r.bm25_rank    != null && <span>BM25 rank: #{r.bm25_rank}</span>}
            {r.semantic_rank != null && <span>Sem rank: #{r.semantic_rank}</span>}
          </div>
        </div>
      ))}
    </div>
  )
}
