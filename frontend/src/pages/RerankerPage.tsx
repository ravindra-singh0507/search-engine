import React, { useState } from 'react'
import { api } from '../api/client'

interface PipelineResult {
  rank: number; doc_id: number; title: string; snippet: string
  final_score: number; bm25_score: number; semantic_score: number
  reranker_score: number; fusion_score: number
}

interface PipelineResponse {
  query: string; total_latency_ms: number; retrieval_count: number
  reranked_count: number; stage_latencies: Record<string, number>
  results: PipelineResult[]
}

export default function RerankerPage() {
  const [query,   setQuery   ] = useState('')
  const [fusion,  setFusion  ] = useState('rrf')
  const [rerank,  setRerank  ] = useState(true)
  const [result,  setResult  ] = useState<PipelineResponse | null>(null)
  const [intent,  setIntent  ] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading ] = useState(false)
  const [error,   setError   ] = useState<string | null>(null)

  const search = async () => {
    if (!query.trim()) return
    setLoading(true); setError(null)
    try {
      const [r, i] = await Promise.all([
        api.rerankSearch(query, 10, fusion, rerank),
        api.queryIntent(query),
      ])
      setResult(r as PipelineResponse)
      setIntent(i)
    } catch (e) { setError(String(e)) }
    finally { setLoading(false) }
  }

  const barColor = (v: number) =>
    v > 0.7 ? '#22c55e' : v > 0.4 ? '#facc15' : '#94a3b8'

  return (
    <div>
      <h1>Multi-Stage Retrieval + Re-ranking</h1>
      <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
        Stage 1: BM25 + Semantic → Stage 2: Fusion → Stage 3: Cross-Encoder Reranker → Final results
      </p>

      {/* Controls */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
        <select value={fusion} onChange={e => setFusion(e.target.value)}
          style={{ background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155',
            borderRadius: 6, padding: '0.5rem', fontSize: '0.875rem' }}>
          {['rrf','combsum','combmnz','weighted','borda'].map(s => (
            <option key={s} value={s}>{s.toUpperCase()}</option>
          ))}
        </select>
        <label style={{ color: '#94a3b8', fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <input type="checkbox" checked={rerank} onChange={e => setRerank(e.target.checked)} />
          Cross-Encoder Reranking
        </label>
      </div>

      <div className="search-bar">
        <input type="text" placeholder="Search with full pipeline…"
          value={query} onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && search()} />
        <button onClick={search}>Search</button>
      </div>

      {loading && <p className="loading">Running pipeline…</p>}
      {error   && <p className="error-msg">{error}</p>}

      {/* Intent badge */}
      {intent && (
        <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <span className={`badge ${
            (intent.intent as string) === 'troubleshooting' ? 'badge-red' :
            (intent.intent as string) === 'transactional'   ? 'badge-green' :
            (intent.intent as string) === 'documentation'   ? 'badge-blue' : 'badge-yellow'
          }`}>{String(intent.intent)}</span>
          <span style={{ color: '#64748b', fontSize: '0.8rem' }}>
            intent confidence: {((intent.confidence as number) * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Pipeline stats */}
      {result && (
        <>
          <div className="card-grid" style={{ marginBottom: '1rem' }}>
            {[
              ['Total latency', result.total_latency_ms.toFixed(1) + ' ms'],
              ['Retrieved', result.retrieval_count],
              ['Reranked', result.reranked_count],
              ['Results', result.results.length],
            ].map(([k, v]) => (
              <div key={String(k)} className="stat-card">
                <div className="value" style={{ fontSize: '1.25rem' }}>{String(v)}</div>
                <div className="label">{String(k)}</div>
              </div>
            ))}
          </div>

          {/* Stage latencies */}
          <div className="card" style={{ marginBottom: '1rem' }}>
            <h2>Stage Latencies</h2>
            {Object.entries(result.stage_latencies).map(([stage, ms]) => (
              <div key={stage} style={{ display: 'flex', justifyContent: 'space-between',
                borderBottom: '1px solid #1e293b', padding: '0.4rem 0', fontSize: '0.875rem' }}>
                <span style={{ color: '#94a3b8' }}>{stage}</span>
                <span style={{ color: '#38bdf8' }}>{ms.toFixed(1)} ms</span>
              </div>
            ))}
          </div>

          {result.results.length === 0 && (
            <div className="empty">No results — try indexing documents first</div>
          )}

          {result.results.map(r => (
            <div key={r.doc_id} className="result-item">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div className="title">#{r.rank} &nbsp; {r.title}</div>
                <div style={{ display: 'flex', gap: '0.4rem', flexShrink: 0 }}>
                  <span className="badge badge-blue">BM25 {r.bm25_score.toFixed(3)}</span>
                  <span className="badge badge-green">cos {r.semantic_score.toFixed(3)}</span>
                  {r.reranker_score > 0 && (
                    <span style={{
                      background: barColor(r.reranker_score),
                      color: '#0f172a', fontWeight: 700,
                      fontSize: '0.7rem', padding: '2px 6px', borderRadius: 9999
                    }}>rerank {r.reranker_score.toFixed(3)}</span>
                  )}
                </div>
              </div>
              <div className="snippet">{r.snippet}</div>
              <div className="meta">
                <span>doc_id: {r.doc_id}</span>
                <span>final: {r.final_score.toFixed(4)}</span>
                <span>fusion: {r.fusion_score.toFixed(4)}</span>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  )
}
