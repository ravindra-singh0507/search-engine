import React, { useEffect, useState } from 'react'
import { api, MetricsSnapshot, StatsResponse } from '../api/client'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer
} from 'recharts'

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null)
  const [stats,   setStats  ] = useState<StatsResponse | null>(null)
  const [history, setHistory] = useState<Array<{ t: number; latency: number }>>([])
  const [error,   setError  ] = useState<string | null>(null)

  const load = async () => {
    try {
      const [m, s] = await Promise.all([api.metrics(), api.stats()])
      setMetrics(m)
      setStats(s)
      setHistory(prev => [
        ...prev.slice(-29),
        { t: Date.now(), latency: m.search_latency.mean },
      ])
    } catch (e: unknown) {
      setError(String(e))
    }
  }

  useEffect(() => {
    load()
    const id = setInterval(load, 5000)
    return () => clearInterval(id)
  }, [])

  if (error)   return <p className="error-msg">{error}</p>
  if (!metrics || !stats) return <p className="loading">Loading metrics…</p>

  const uptimeH = (metrics.uptime_seconds / 3600).toFixed(1)

  return (
    <div>
      <h1>Performance Metrics</h1>

      {/* Index stats row */}
      <div className="card-grid">
        {[
          ['Documents',  stats.total_documents.toLocaleString()],
          ['Terms',      stats.total_terms.toLocaleString()],
          ['Postings',   stats.total_postings.toLocaleString()],
          ['Avg Doc Len',stats.avg_document_length + ' tokens'],
          ['Uptime',     uptimeH + ' h'],
          ['Crawled',    stats.total_crawled_pages.toLocaleString()],
        ].map(([label, value]) => (
          <div key={label as string} className="stat-card">
            <div className="value">{value}</div>
            <div className="label">{label}</div>
          </div>
        ))}
      </div>

      {/* Counters */}
      <div className="card-grid">
        {[
          ['Searches',    metrics.search_requests_total,    'badge-blue'],
          ['Index Ops',   metrics.index_operations_total,   'badge-green'],
          ['Crawl Pages', metrics.crawl_pages_total,        'badge-green'],
          ['Slow Queries',metrics.slow_queries_total,       'badge-yellow'],
        ].map(([label, value, badge]) => (
          <div key={label as string} className="stat-card">
            <div><span className={`badge ${badge}`}>{value as number}</span></div>
            <div className="label" style={{ marginTop: '0.5rem' }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Latency history chart */}
      {history.length > 1 && (
        <div className="card">
          <h2>Search Latency — rolling 30 samples</h2>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="t" hide />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} unit=" ms" />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                formatter={(v: unknown) => [`${Number(v).toFixed(2)} ms`, 'Latency']}
              />
              <Line type="monotone" dataKey="latency" stroke="#38bdf8" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Cache stats */}
      <div className="card">
        <h2>Cache</h2>
        <div className="metrics-grid">
          {[
            ['Hit Rate',    (metrics.cache.hit_rate * 100).toFixed(1) + '%'],
            ['Cache Size',  metrics.cache.size + ' / ' + metrics.cache.capacity],
          ].map(([k, v]) => (
            <div key={k as string} className="stat-card">
              <div className="value">{v}</div>
              <div className="label">{k}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Latency detail */}
      <div className="card">
        <h2>Search Latency Detail</h2>
        <table>
          <tbody>
            {Object.entries(metrics.search_latency).map(([k, v]) => (
              <tr key={k}>
                <td style={{ color: '#64748b' }}>{k}</td>
                <td>{typeof v === 'number' ? v.toFixed(2) + ' ms' : JSON.stringify(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
