import React, { useEffect, useState } from 'react'
import { api } from '../api/client'

interface CrawlStatus {
  status: string
  seeds?: string[]
  max_depth?: number
  max_pages?: number
  stats?: {
    pages_crawled: number
    pages_indexed: number
    pages_failed: number
    pages_skipped_robots: number
    links_discovered: number
    duration_seconds: number
  }
}

export default function CrawlerPage() {
  const [status,    setStatus  ] = useState<CrawlStatus | null>(null)
  const [seeds,     setSeeds   ] = useState('')
  const [maxDepth,  setMaxDepth] = useState(2)
  const [maxPages,  setMaxPages] = useState(20)
  const [submitting,setSubmit  ] = useState(false)
  const [error,     setError   ] = useState<string | null>(null)
  const [msg,       setMsg     ] = useState<string | null>(null)

  const loadStatus = () =>
    api.crawlStatus()
      .then(s => setStatus(s as CrawlStatus))
      .catch(() => {})

  useEffect(() => {
    loadStatus()
    const id = setInterval(loadStatus, 3000)
    return () => clearInterval(id)
  }, [])

  const startCrawl = async () => {
    const urls = seeds.split('\n').map(s => s.trim()).filter(Boolean)
    if (!urls.length) { setError('Enter at least one seed URL'); return }
    setSubmit(true); setError(null); setMsg(null)
    try {
      await fetch('/api/crawl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed_urls: urls, max_depth: maxDepth, max_pages: maxPages }),
      })
      setMsg('Crawl started!')
      loadStatus()
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setSubmit(false)
    }
  }

  const s = status?.stats

  return (
    <div>
      <h1>Web Crawler</h1>

      {/* Start crawl form */}
      <div className="card">
        <h2>Start New Crawl</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <textarea
            rows={3}
            placeholder="One seed URL per line&#10;e.g. https://docs.python.org/3/tutorial/"
            value={seeds}
            onChange={e => setSeeds(e.target.value)}
            style={{
              background: '#0f172a', border: '1px solid #334155',
              borderRadius: '0.5rem', padding: '0.75rem',
              color: '#e2e8f0', fontFamily: 'monospace', fontSize: '0.875rem',
            }}
          />
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <label style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
              Max Depth&nbsp;
              <input type="number" min={1} max={5} value={maxDepth}
                onChange={e => setMaxDepth(Number(e.target.value))}
                style={{ width: '4rem', background: '#0f172a', border: '1px solid #334155',
                  borderRadius: '0.25rem', padding: '0.25rem 0.5rem', color: '#e2e8f0' }} />
            </label>
            <label style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
              Max Pages&nbsp;
              <input type="number" min={1} max={500} value={maxPages}
                onChange={e => setMaxPages(Number(e.target.value))}
                style={{ width: '5rem', background: '#0f172a', border: '1px solid #334155',
                  borderRadius: '0.25rem', padding: '0.25rem 0.5rem', color: '#e2e8f0' }} />
            </label>
            <button
              onClick={startCrawl}
              disabled={submitting || status?.status === 'running'}
              style={{
                background: '#38bdf8', color: '#0f172a', border: 'none',
                borderRadius: '0.5rem', padding: '0.5rem 1.25rem',
                fontWeight: 600, cursor: 'pointer', opacity: submitting ? 0.6 : 1,
              }}
            >
              {submitting ? 'Starting…' : 'Start Crawl'}
            </button>
          </div>
          {error && <p className="error-msg">{error}</p>}
          {msg   && <p style={{ color: '#22c55e' }}>{msg}</p>}
        </div>
      </div>

      {/* Status */}
      <div className="card">
        <h2>Crawler Status</h2>
        {!status
          ? <p className="loading">Loading…</p>
          : (
            <>
              <p style={{ marginBottom: '1rem' }}>
                Status:&nbsp;
                <span className={`badge ${status.status === 'running' ? 'badge-green' : status.status === 'complete' ? 'badge-blue' : 'badge-yellow'}`}>
                  {status.status}
                </span>
              </p>

              {s && (
                <div className="card-grid">
                  {[
                    ['Crawled',  s.pages_crawled],
                    ['Indexed',  s.pages_indexed],
                    ['Failed',   s.pages_failed],
                    ['Skipped (robots)', s.pages_skipped_robots],
                    ['Links found', s.links_discovered],
                    ['Duration', s.duration_seconds.toFixed(1) + ' s'],
                  ].map(([label, value]) => (
                    <div key={label as string} className="stat-card">
                      <div className="value">{value}</div>
                      <div className="label">{label}</div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )
        }
      </div>
    </div>
  )
}
