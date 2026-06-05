import { useState, useEffect } from 'react'

const API = 'http://localhost:8000'

interface StepResult {
  task_id: string
  agent_type: string
  status: string
  output: any
  confidence: number
  latency_ms: number
  error: string | null
}

interface ResearchResult {
  session_id: string
  run_id: string
  status: string
  total_steps: number
  success_count: number
  failure_count: number
  total_latency_ms: number
  report: any | null
  step_results: Record<string, StepResult>
}

const statusColor = (s: string) => {
  if (s === 'completed' || s === 'done') return 'text-green-600 bg-green-50'
  if (s === 'running' || s === 'pending') return 'text-yellow-600 bg-yellow-50'
  return 'text-red-600 bg-red-50'
}

export default function AgentDashboardPage() {
  const [goal, setGoal]               = useState('')
  const [workflow, setWorkflow]       = useState('investigation')
  const [parallel, setParallel]       = useState(false)
  const [loading, setLoading]         = useState(false)
  const [result, setResult]           = useState<ResearchResult | null>(null)
  const [error, setError]             = useState('')
  const [workflows, setWorkflows]     = useState<{name: string; description: string}[]>([])
  const [metrics, setMetrics]         = useState<any>(null)
  const [tab, setTab]                 = useState<'research' | 'metrics' | 'tools'>('research')

  useEffect(() => {
    fetch(`${API}/research/workflows`)
      .then(r => r.json())
      .then(d => setWorkflows(d.workflows || []))
      .catch(() => {})
  }, [])

  const runResearch = async () => {
    if (!goal.trim() || loading) return
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const res = await fetch(`${API}/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal, workflow, parallel, params: {} }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e.message)
    }
    setLoading(false)
  }

  const loadMetrics = async () => {
    try {
      const res = await fetch(`${API}/research/metrics`)
      const data = await res.json()
      setMetrics(data)
    } catch (e: any) {
      setError(e.message)
    }
  }

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Agent Research Dashboard</h1>
        <p className="text-sm text-gray-500">
          Agentic retrieval with planning, evidence gathering, critique, and synthesis
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b pb-2">
        {(['research', 'metrics', 'tools'] as const).map(t => (
          <button
            key={t}
            onClick={() => { setTab(t); if (t === 'metrics') loadMetrics(); }}
            className={`px-4 py-1.5 text-sm rounded-t-lg font-medium ${
              tab === t ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Research Tab */}
      {tab === 'research' && (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-3 p-4 bg-gray-50 rounded-xl border">
            <input
              type="text"
              value={goal}
              onChange={e => setGoal(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && runResearch()}
              placeholder="Enter research goal..."
              className="flex-1 min-w-[300px] border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={loading}
            />
            <select
              value={workflow}
              onChange={e => setWorkflow(e.target.value)}
              className="border rounded-lg px-3 py-2 text-sm bg-white"
            >
              {workflows.map(w => (
                <option key={w.name} value={w.name}>{w.name}</option>
              ))}
            </select>
            <label className="flex items-center gap-1.5 text-sm">
              <input type="checkbox" checked={parallel} onChange={e => setParallel(e.target.checked)} />
              Parallel
            </label>
            <button
              onClick={runResearch}
              disabled={loading || !goal.trim()}
              className="px-5 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white rounded-lg text-sm font-medium"
            >
              {loading ? 'Running...' : 'Research'}
            </button>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              Error: {error}
            </div>
          )}

          {result && (
            <div className="space-y-4">
              {/* Summary */}
              <div className="p-4 bg-white border rounded-xl shadow-sm">
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">Research Result</h2>
                    <p className="text-xs text-gray-400 font-mono mt-1">
                      Session: {result.session_id.slice(0, 12)}... | Run: {result.run_id.slice(0, 12)}...
                    </p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${statusColor(result.status)}`}>
                    {result.status}
                  </span>
                </div>
                <div className="flex gap-6 mt-3 text-sm text-gray-600">
                  <span>Steps: {result.success_count}/{result.total_steps} succeeded</span>
                  {result.failure_count > 0 && (
                    <span className="text-red-600">{result.failure_count} failed</span>
                  )}
                  <span>{result.total_latency_ms.toFixed(0)} ms</span>
                </div>
              </div>

              {/* Step Results */}
              <div className="grid gap-3">
                <h3 className="text-sm font-semibold text-gray-700">Agent Steps</h3>
                {Object.entries(result.step_results).map(([stepId, step]) => (
                  <StepCard key={stepId} stepId={stepId} step={step} />
                ))}
              </div>

              {/* Report */}
              {result.report && (
                <div className="p-4 bg-white border rounded-xl shadow-sm">
                  <h3 className="text-sm font-semibold text-gray-700 mb-2">Generated Report</h3>
                  <div className="prose prose-sm max-w-none text-gray-800 whitespace-pre-wrap">
                    {result.report.full_report || result.report.summary || JSON.stringify(result.report, null, 2)}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Metrics Tab */}
      {tab === 'metrics' && (
        <div className="space-y-4">
          <button onClick={loadMetrics}
            className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm">
            Refresh Metrics
          </button>
          {metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard label="Agent Executions" value={metrics.agent_executions} />
              <MetricCard label="Agent Successes" value={metrics.agent_successes} />
              <MetricCard label="Agent Failures" value={metrics.agent_failures} />
              <MetricCard label="Workflow Runs" value={metrics.workflow_runs} />
              <MetricCard label="Workflow Completions" value={metrics.workflow_completions} />
              <MetricCard label="Avg Agent Latency"
                value={`${metrics.agent_latency?.mean?.toFixed(0) || 0} ms`} />
              <MetricCard label="Avg Workflow Latency"
                value={`${metrics.workflow_latency?.mean?.toFixed(0) || 0} ms`} />
            </div>
          )}
        </div>
      )}

      {/* Tools Tab */}
      {tab === 'tools' && <ToolsPanel />}
    </div>
  )
}

function StepCard({ stepId, step }: { stepId: string; step: StepResult }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-white border rounded-lg p-3 text-sm">
      <div className="flex items-center justify-between cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-gray-400">{stepId}</span>
          <span className="font-medium text-gray-800">{step.agent_type}</span>
          <span className={`px-2 py-0.5 rounded-full text-xs ${statusColor(step.status)}`}>
            {step.status}
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-gray-500">
          <span>Confidence: {(step.confidence * 100).toFixed(0)}%</span>
          <span>{step.latency_ms?.toFixed(0)} ms</span>
          <span>{expanded ? '▲' : '▼'}</span>
        </div>
      </div>
      {expanded && step.output && (
        <pre className="mt-2 p-2 bg-gray-50 rounded text-xs overflow-auto max-h-64">
          {typeof step.output === 'string' ? step.output : JSON.stringify(step.output, null, 2)}
        </pre>
      )}
      {expanded && step.error && (
        <p className="mt-2 text-xs text-red-600">Error: {step.error}</p>
      )}
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: any }) {
  return (
    <div className="p-4 bg-white border rounded-xl text-center">
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      <p className="text-xs text-gray-500 mt-1">{label}</p>
    </div>
  )
}

function ToolsPanel() {
  const [tools, setTools] = useState<any[]>([])

  useEffect(() => {
    fetch(`${API}/tools`)
      .then(r => r.json())
      .then(d => setTools(d.tools || []))
      .catch(() => {})
  }, [])

  return (
    <div className="grid gap-3">
      <h3 className="text-sm font-semibold text-gray-700">Available Tools</h3>
      {tools.map(t => (
        <div key={t.name} className="p-3 bg-white border rounded-lg">
          <div className="flex justify-between">
            <span className="font-medium text-gray-800">{t.name}</span>
            <span className="text-xs text-gray-400">MCP-compatible</span>
          </div>
          <p className="text-xs text-gray-600 mt-1">{t.description}</p>
        </div>
      ))}
    </div>
  )
}
