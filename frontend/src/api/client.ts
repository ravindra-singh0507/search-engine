/* API client — thin wrapper around fetch() */

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${path}`)
  return r.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${path}`)
  return r.json() as Promise<T>
}

async function del<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path, { method: 'DELETE' })
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${path}`)
  return r.json() as Promise<T>
}

// ── Phase 1-3 types ────────────────────────────────────────────────────────

export interface SearchResultItem {
  rank: number; doc_id: number; score: number; bm25_score: number
  title_score: number; title: string; snippet: string
  term_scores: Record<string, number>
}

export interface SearchResponse {
  query: string; corrected_query: string | null; expanded_terms: string[]
  total_matches: number; search_time_ms: number; cache_hit: boolean
  log_id: number | null; results: SearchResultItem[]
}

export interface AutocompleteSuggestion { suggestion: string; frequency: number }

export interface StatsResponse {
  total_documents: number; total_terms: number; total_postings: number
  total_crawled_pages: number; avg_document_length: number
}

export interface MetricsSnapshot {
  uptime_seconds: number; search_requests_total: number
  index_operations_total: number; crawl_pages_total: number
  slow_queries_total: number; cache_hit_rate: number
  semantic_searches_total?: number; hybrid_searches_total?: number
  embedding_cache_hit_rate?: number
  search_latency: { count: number; mean: number }
  embedding_latency?: { count: number; mean: number }
  semantic_search_latency?: { count: number; mean: number }
  cache: { hit_rate: number; size: number; capacity: number }
}

export interface AnalyticsDashboard {
  click_through_rate: { total_searches: number; searches_with_clicks: number; click_through_rate: number }
  avg_click_position: number
  top_queries: Array<{ query: string; total_searches: number; avg_latency_ms: number }>
  failed_queries: Array<{ query: string; zero_result_searches: number; total_searches: number }>
  search_volume_24h: Array<{ hour: string; searches: number }>
}

// ── Phase 4 types ──────────────────────────────────────────────────────────

export interface SemanticResultItem {
  rank: number; doc_id: number; chunk_id: string; title: string
  snippet: string; chunk_text: string; semantic_score: number
}

export interface SemanticSearchResponse {
  query: string; model: string; search_time_ms: number
  total_results: number; results: SemanticResultItem[]
}

export interface HybridResultItem {
  rank: number; doc_id: number; title: string; snippet: string
  fusion_score: number; bm25_score: number; bm25_rank: number | null
  semantic_score: number; semantic_rank: number | null
}

export interface HybridSearchResponse {
  query: string; fusion_strategy: string; search_time_ms: number
  bm25_results: number; semantic_results: number
  total_results: number; results: HybridResultItem[]
}

export interface EmbeddingStats {
  model_name: string; dimension: number
  vector_store: { total_vectors: number; dimension: number }
  semantic_stats: { total_chunks: number; embedded_chunks: number; unembedded_docs: number }
  embedding_cache: { hits: number; misses: number; hit_rate: number; db_entries: number }
  jobs_running: boolean
  recent_jobs: Array<{ job_id: number; status: string; chunks_processed: number; created_at: string }>
}

export interface EvaluationResult {
  eval_queries: number; systems: string[]
  results: Record<string, Record<string, number>>
}

// ── API object ─────────────────────────────────────────────────────────────

export const api = {
  // Phase 1-3
  search: (q: string, topK = 10) =>
    get<SearchResponse>(`/search?q=${encodeURIComponent(q)}&top_k=${topK}`),
  autocomplete: (prefix: string) =>
    get<{ prefix: string; suggestions: AutocompleteSuggestion[] }>(
      `/autocomplete?q=${encodeURIComponent(prefix)}`),
  spellcheck: (q: string) =>
    get<{ input: string; is_known: boolean; suggestions: unknown[] }>(
      `/spellcheck?q=${encodeURIComponent(q)}`),
  stats:       () => get<StatsResponse>('/stats'),
  metrics:     () => get<MetricsSnapshot>('/metrics/snapshot'),
  analytics:   () => get<AnalyticsDashboard>('/analytics/dashboard'),
  crawlStatus: () => get<Record<string, unknown>>('/crawl/status'),
  recordClick: (log_id: number, doc_id: number, position: number) =>
    post('/search/click', { log_id, doc_id, position }),

  // Phase 4
  semanticSearch:      (q: string, topK = 10) =>
    get<SemanticSearchResponse>(
      `/semantic-search?q=${encodeURIComponent(q)}&top_k=${topK}`),
  hybridSearch:        (q: string, topK = 10) =>
    get<HybridSearchResponse>(
      `/hybrid-search?q=${encodeURIComponent(q)}&top_k=${topK}`),
  embeddingsStats:     () => get<EmbeddingStats>('/embeddings/stats'),
  reindex:             (force = false) =>
    post('/embeddings/reindex', { force }),
  clearEmbeddingCache: () => del<{ status: string }>('/embeddings/cache'),
  evaluation:          (systems = 'bm25,semantic,hybrid') =>
    get<EvaluationResult>(`/evaluation?systems=${encodeURIComponent(systems)}`),
  hybridExplain:       (q: string, docId: number) =>
    get<Record<string, unknown>>(
      `/hybrid-search/explain?q=${encodeURIComponent(q)}&doc_id=${docId}`),
  vectorStoreStats:    () => get<Record<string, unknown>>('/vector-store/stats'),

  // Phase 5
  rerankSearch:      (q: string, topK = 10, fusion = 'rrf', rerank = true) =>
    get<Record<string, unknown>>(
      `/rerank-search?q=${encodeURIComponent(q)}&top_k=${topK}&fusion=${fusion}&rerank=${rerank}`),
  rerankExplain:     (q: string, docId: number) =>
    get<Record<string, unknown>>(
      `/rerank/explain?q=${encodeURIComponent(q)}&doc_id=${docId}`),
  fusionCompare:     (q: string, topK = 10) =>
    get<Record<string, unknown>>(
      `/fusion/compare?q=${encodeURIComponent(q)}&top_k=${topK}`),
  queryIntent:       (q: string) =>
    get<Record<string, unknown>>(`/query/intent?q=${encodeURIComponent(q)}`),
  intentDistribution: () =>
    get<Record<string, unknown>>('/query/intents/distribution'),
  getExperiments:    () => get<Record<string, unknown>>('/experiments'),
  runExperiment:     (name: string, systems: string) =>
    post<Record<string, unknown>>(`/experiments/run?name=${encodeURIComponent(name)}&systems=${encodeURIComponent(systems)}`, {}),
  rankingFeatures:   (q: string, docId: number) =>
    get<Record<string, unknown>>(
      `/ranking/features?q=${encodeURIComponent(q)}&doc_id=${docId}`),
  pipelineStats:     () => get<Record<string, unknown>>('/retrieval-pipeline/stats'),
}
