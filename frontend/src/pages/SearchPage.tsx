import React, { useEffect, useRef, useState } from 'react'
import { api, SearchResponse, AutocompleteSuggestion } from '../api/client'

export default function SearchPage() {
  const [query, setQuery]       = useState('')
  const [result, setResult]     = useState<SearchResponse | null>(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<AutocompleteSuggestion[]>([])
  const [showAC, setShowAC]     = useState(false)
  const acTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Autocomplete debounce
  useEffect(() => {
    if (!query || query.length < 2) { setSuggestions([]); return }
    if (acTimer.current) clearTimeout(acTimer.current)
    acTimer.current = setTimeout(async () => {
      try {
        const res = await api.autocomplete(query)
        setSuggestions(res.suggestions)
      } catch { setSuggestions([]) }
    }, 200)
  }, [query])

  const handleSearch = async (q = query) => {
    if (!q.trim()) return
    setLoading(true)
    setError(null)
    setShowAC(false)
    try {
      const res = await api.search(q)
      setResult(res)
    } catch (e: unknown) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleSuggestionClick = (s: string) => {
    setQuery(s)
    setShowAC(false)
    handleSearch(s)
  }

  const handleClick = (docId: number, rank: number) => {
    if (result?.log_id) {
      api.recordClick(result.log_id, docId, rank - 1).catch(() => {})
    }
  }

  // Render snippet with **bold** markers converted to <strong>
  const renderSnippet = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/)
    return parts.map((p, i) =>
      p.startsWith('**') && p.endsWith('**')
        ? <strong key={i}>{p.slice(2, -2)}</strong>
        : p
    )
  }

  return (
    <div>
      <h1>Search</h1>

      {/* Search bar */}
      <div className="search-bar">
        <input
          type="text"
          placeholder='Try "python AND backend" or title:machine'
          value={query}
          onChange={e => { setQuery(e.target.value); setShowAC(true) }}
          onKeyDown={e => { if (e.key === 'Enter') handleSearch(); if (e.key === 'Escape') setShowAC(false) }}
          onFocus={() => setShowAC(true)}
          autoComplete="off"
        />
        <button onClick={() => handleSearch()}>Search</button>

        {/* Autocomplete dropdown */}
        {showAC && suggestions.length > 0 && (
          <div className="autocomplete-dropdown">
            {suggestions.map(s => (
              <div key={s.suggestion} className="autocomplete-item"
                onMouseDown={() => handleSuggestionClick(s.suggestion)}>
                <span>{s.suggestion}</span>
                <span className="freq">×{s.frequency}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {loading && <p className="loading">Searching…</p>}
      {error   && <p className="error-msg">{error}</p>}

      {result && (
        <>
          {/* Spell correction banner */}
          {result.corrected_query && (
            <div className="correction-banner">
              Showing results for <strong>{result.corrected_query}</strong>
              {' '}&mdash; did you mean that?
            </div>
          )}

          {/* Meta */}
          <p style={{ color: '#64748b', marginBottom: '1rem', fontSize: '0.875rem' }}>
            {result.total_matches} matches &nbsp;·&nbsp; {result.search_time_ms} ms
            {result.cache_hit && <span style={{ color: '#22c55e' }}> &nbsp;· cached</span>}
            {result.expanded_terms.length > 0 &&
              <span> &nbsp;· expanded: {result.expanded_terms.join(', ')}</span>}
          </p>

          {/* Results */}
          {result.results.length === 0
            ? <div className="empty">No results found for "{result.query}"</div>
            : result.results.map(r => (
              <div key={r.doc_id} className="result-item"
                onClick={() => handleClick(r.doc_id, r.rank)}>
                <div className="title">#{r.rank} &nbsp; {r.title}</div>
                <div className="snippet">{renderSnippet(r.snippet)}</div>
                <div className="meta">
                  <span>doc_id: {r.doc_id}</span>
                  <span>score: {r.score.toFixed(4)}</span>
                  <span>bm25: {r.bm25_score.toFixed(4)}</span>
                  {r.title_score > 0 && <span>title↑ {r.title_score.toFixed(3)}</span>}
                </div>
              </div>
            ))
          }
        </>
      )}
    </div>
  )
}
