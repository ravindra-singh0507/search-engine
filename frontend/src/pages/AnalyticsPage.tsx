import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts'
import { api, AnalyticsDashboard } from '../api/client'

export default function AnalyticsPage() {
  const [data, setData]   = useState<AnalyticsDashboard | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.analytics()
      .then(setData)
      .catch(e => setError(String(e)))
  }, [])

  if (error) return <p className="error-msg">{error}</p>
  if (!data) return <p className="loading">Loading analytics…</p>

  const { click_through_rate: ctr, avg_click_position, top_queries, failed_queries, search_volume_24h } = data

  return (
    <div>
      <h1>Analytics</h1>

      {/* KPI row */}
      <div className="card-grid">
        <div className="stat-card">
          <div className="value">{ctr.total_searches.toLocaleString()}</div>
          <div className="label">Total Searches</div>
        </div>
        <div className="stat-card">
          <div className="value">{(ctr.click_through_rate * 100).toFixed(1)}%</div>
          <div className="label">Click-Through Rate</div>
        </div>
        <div className="stat-card">
          <div className="value">{avg_click_position.toFixed(1)}</div>
          <div className="label">Avg Click Position</div>
        </div>
        <div className="stat-card">
          <div className="value">{ctr.searches_with_clicks.toLocaleString()}</div>
          <div className="label">Searches with Clicks</div>
        </div>
      </div>

      {/* Search volume chart */}
      {search_volume_24h.length > 0 && (
        <div className="card">
          <h2>Search Volume (24 h)</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={search_volume_24h}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="hour" tick={{ fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => v.slice(11, 16)} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                labelStyle={{ color: '#94a3b8' }} />
              <Bar dataKey="searches" fill="#38bdf8" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Top queries */}
      <div className="card">
        <h2>Top Queries</h2>
        {top_queries.length === 0
          ? <p className="empty">No query data yet</p>
          : (
            <table>
              <thead>
                <tr><th>Query</th><th>Searches</th><th>Avg Latency</th></tr>
              </thead>
              <tbody>
                {top_queries.map(q => (
                  <tr key={q.query}>
                    <td><code>{q.query}</code></td>
                    <td>{q.total_searches}</td>
                    <td>{q.avg_latency_ms.toFixed(1)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        }
      </div>

      {/* Failed queries */}
      <div className="card">
        <h2>Failed Queries (Zero Results)</h2>
        {failed_queries.length === 0
          ? <p className="empty">No failed queries</p>
          : (
            <table>
              <thead>
                <tr><th>Query</th><th>Zero-Result Hits</th><th>Total Searches</th></tr>
              </thead>
              <tbody>
                {failed_queries.map(q => (
                  <tr key={q.query}>
                    <td><code>{q.query}</code></td>
                    <td><span className="badge badge-red">{q.zero_result_searches}</span></td>
                    <td>{q.total_searches}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        }
      </div>
    </div>
  )
}
