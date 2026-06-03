import React, { useEffect, useState } from 'react'
import { api } from '../api/client'

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<unknown[]>([])
  const [runs,        setRuns       ] = useState<unknown[]>([])
  const [name,        setName       ] = useState('')
  const [systems,     setSystems    ] = useState('bm25,semantic,hybrid')
  const [running,     setRunning    ] = useState(false)
  const [runResult,   setRunResult  ] = useState<Record<string,unknown> | null>(null)
  const [error,       setError      ] = useState<string | null>(null)
  const [intents,     setIntents    ] = useState<unknown[]>([])

  useEffect(() => {
    api.getExperiments().then(d => {
      setExperiments((d as Record<string, unknown[]>).experiments || [])
      setRuns((d as Record<string, unknown[]>).runs || [])
    }).catch(() => {})
    api.intentDistribution().then(d => {
      setIntents((d as Record<string, unknown[]>).distribution || [])
    }).catch(() => {})
  }, [])

  const runExp = async () => {
    if (!name.trim()) { setError('Enter experiment name'); return }
    setRunning(true); setError(null)
    try {
      const r = await api.runExperiment(name, systems)
      setRunResult(r as Record<string, unknown>)
      const d = await api.getExperiments()
      setRuns((d as Record<string, unknown[]>).runs || [])
    } catch (e) { setError(String(e)) }
    finally { setRunning(false) }
  }

  return (
    <div>
      <h1>Retrieval Experiments</h1>
      <p style={{ color: '#64748b', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
        A/B compare BM25, Semantic, and Hybrid retrieval against the evaluation dataset.
      </p>

      {/* Run form */}
      <div className="card">
        <h2>Run New Experiment</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <input type="text" placeholder="Experiment name…"
            value={name} onChange={e => setName(e.target.value)}
            style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6,
              padding: '0.6rem 1rem', color: '#e2e8f0' }} />
          <input type="text" placeholder="Systems (comma-separated): bm25,semantic,hybrid"
            value={systems} onChange={e => setSystems(e.target.value)}
            style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6,
              padding: '0.6rem 1rem', color: '#e2e8f0' }} />
          <button onClick={runExp} disabled={running}
            style={{ background: '#38bdf8', color: '#0f172a', border: 'none',
              borderRadius: 6, padding: '0.6rem 1.5rem', fontWeight: 700,
              cursor: 'pointer', width: 'fit-content' }}>
            {running ? 'Running…' : 'Run Experiment'}
          </button>
          {error && <p className="error-msg">{error}</p>}
        </div>
      </div>

      {/* Run result */}
      {runResult && (
        <div className="card">
          <h2>Latest Run: {String(runResult.run_id)}</h2>
          <p style={{ color: '#64748b', fontSize: '0.875rem', marginBottom: '1rem' }}>
            {String(runResult.query_count)} queries · {String(runResult.latency_ms)} ms
          </p>
          <table>
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
              {Object.entries(runResult.metrics as Record<string, number> || {}).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td style={{ color: '#38bdf8' }}>{v.toFixed(4)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Past runs */}
      {runs.length > 0 && (
        <div className="card">
          <h2>Past Runs</h2>
          <table>
            <thead><tr><th>Run ID</th><th>Experiment</th><th>Queries</th><th>Latency</th></tr></thead>
            <tbody>
              {(runs as Record<string,unknown>[]).map(r => (
                <tr key={String(r.run_id)}>
                  <td><code style={{ fontSize: '0.75rem' }}>{String(r.run_id)}</code></td>
                  <td>{String(r.experiment_id)}</td>
                  <td>{String(r.query_count)}</td>
                  <td>{String(r.latency_ms)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Intent distribution */}
      {intents.length > 0 && (
        <div className="card">
          <h2>Query Intent Distribution</h2>
          <table>
            <thead><tr><th>Intent</th><th>Count</th><th>Avg Confidence</th></tr></thead>
            <tbody>
              {(intents as Record<string,unknown>[]).map(i => (
                <tr key={String(i.intent)}>
                  <td>
                    <span className={`badge ${
                      i.intent === 'troubleshooting' ? 'badge-red' :
                      i.intent === 'transactional'   ? 'badge-green' :
                      i.intent === 'documentation'   ? 'badge-blue' : 'badge-yellow'
                    }`}>{String(i.intent)}</span>
                  </td>
                  <td>{String(i.cnt)}</td>
                  <td>{Number(i.avg_confidence).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
