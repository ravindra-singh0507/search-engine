import React from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import SearchPage        from './pages/SearchPage'
import AnalyticsPage     from './pages/AnalyticsPage'
import MetricsPage       from './pages/MetricsPage'
import CrawlerPage       from './pages/CrawlerPage'
import SemanticSearchPage from './pages/SemanticSearchPage'
import HybridSearchPage   from './pages/HybridSearchPage'
import './index.css'

const NAV = [
  ['/',         'Search'],
  ['/semantic', 'Semantic'],
  ['/hybrid',   'Hybrid'],
  ['/analytics','Analytics'],
  ['/metrics',  'Metrics'],
  ['/crawler',  'Crawler'],
] as const

export default function App() {
  return (
    <BrowserRouter>
      <nav>
        <span className="brand">⚡ Search Engine</span>
        {NAV.map(([to, label]) => (
          <NavLink key={to} to={to} end={to === '/'}
            className={({ isActive }) => isActive ? 'active' : ''}>
            {label}
          </NavLink>
        ))}
      </nav>
      <main>
        <Routes>
          <Route path="/"          element={<SearchPage />} />
          <Route path="/semantic"  element={<SemanticSearchPage />} />
          <Route path="/hybrid"    element={<HybridSearchPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/metrics"   element={<MetricsPage />} />
          <Route path="/crawler"   element={<CrawlerPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
