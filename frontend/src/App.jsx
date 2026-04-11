import { useState, useRef, useEffect, useCallback } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const LAYER_COLORS = [
  '#2e7d32', '#1565c0', '#e65100', '#7b1fa2', '#c62828',
  '#00838f', '#9e9d24', '#ad1457', '#4527a0', '#ef6c00',
]

const NODE_W = 160
const NODE_H = 44
const GAP_X = 24
const GAP_Y = 90
const PAD = 40

function parseMermaidGraph(code) {
  const nodes = {}  // id -> { id, label }
  const edges = []  // [fromId, toId]

  for (const line of code.split('\n')) {
    // Match: A["label"] --> B["label"]  or  A --> B  etc.
    const edgeMatch = line.match(/^\s*(\w+)(?:\s*\[\"([^"]*)\"\])?\s*--+>?\s*(\w+)(?:\s*\[\"([^"]*)\"\])?/)
    if (edgeMatch) {
      const [, id1, label1, id2, label2] = edgeMatch
      if (!nodes[id1]) nodes[id1] = { id: id1, label: label1 || id1 }
      else if (label1) nodes[id1].label = label1
      if (!nodes[id2]) nodes[id2] = { id: id2, label: label2 || id2 }
      else if (label2) nodes[id2].label = label2
      edges.push([id1, id2])
      continue
    }
    // Match standalone node definition: A["label"]
    const nodeMatch = line.match(/^\s*(\w+)\s*\[\"([^"]*)\"\]/)
    if (nodeMatch && !nodes[nodeMatch[1]]) {
      nodes[nodeMatch[1]] = { id: nodeMatch[1], label: nodeMatch[2] }
    }
  }
  return { nodes, edges }
}

function layoutGraph(nodes, edges) {
  const ids = Object.keys(nodes)
  if (ids.length === 0) return { positioned: [], layoutEdges: [], width: 0, height: 0 }

  // Build adjacency (parent -> children) for BFS depth
  const adj = {}
  const targets = new Set()
  for (const [from, to] of edges) {
    if (!adj[from]) adj[from] = []
    adj[from].push(to)
    targets.add(to)
  }
  const roots = ids.filter((id) => !targets.has(id))
  if (roots.length === 0) roots.push(ids[0])

  // BFS - assign max depth (longest path from any root)
  const depth = {}
  for (const r of roots) depth[r] = 0
  let changed = true
  while (changed) {
    changed = false
    for (const [from, to] of edges) {
      const d = (depth[from] ?? -1) + 1
      if (d > (depth[to] ?? -1)) {
        depth[to] = d
        changed = true
      }
    }
  }
  // Assign any orphans
  for (const id of ids) if (!(id in depth)) depth[id] = 0

  // Group by layer
  const layers = {}
  for (const id of ids) {
    const d = depth[id]
    if (!layers[d]) layers[d] = []
    layers[d].push(id)
  }
  const maxLayer = Math.max(...Object.keys(layers).map(Number))

  // Position nodes: split each layer into multiple sub-rows (max ~8 per row)
  const MAX_PER_ROW = 8
  const ROW_GAP = 10
  const positioned = []
  const posMap = {} // id -> {x, y}
  let currentY = 0
  for (let layer = 0; layer <= maxLayer; layer++) {
    const group = layers[layer] || []
    if (group.length === 0) continue
    const numRows = Math.ceil(group.length / MAX_PER_ROW)
    const subRows = []
    for (let r = 0; r < numRows; r++) {
      subRows.push(group.slice(r * MAX_PER_ROW, (r + 1) * MAX_PER_ROW))
    }
    const placeRow = (row, y) => {
      const totalW = row.length * NODE_W + (row.length - 1) * GAP_X
      const startX = -totalW / 2 + NODE_W / 2
      row.forEach((id, i) => {
        const x = startX + i * (NODE_W + GAP_X)
        posMap[id] = { x, y }
        positioned.push({
          ...nodes[id], x, y,
          depth: layer,
          color: LAYER_COLORS[layer % LAYER_COLORS.length],
        })
      })
    }
    subRows.forEach((row, ri) => {
      placeRow(row, currentY + ri * (NODE_H + ROW_GAP))
    })
    currentY += subRows.length * (NODE_H + ROW_GAP) + GAP_Y
  }

  // Build edge lines (keep IDs for hover logic)
  const layoutEdges = edges.map(([from, to]) => {
    const f = posMap[from]
    const t = posMap[to]
    if (!f || !t) return null
    return { fromId: from, toId: to, from: f, to: t }
  }).filter(Boolean)

  // Build bidirectional adjacency for hover traversal
  const fwd = {}  // id -> [ids reachable via edges]
  const bwd = {}  // id -> [ids reachable via reverse edges]
  for (const [from, to] of edges) {
    if (!fwd[from]) fwd[from] = []
    fwd[from].push(to)
    if (!bwd[to]) bwd[to] = []
    bwd[to].push(from)
  }

  // Compute bounding box
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const n of positioned) {
    minX = Math.min(minX, n.x - NODE_W / 2)
    maxX = Math.max(maxX, n.x + NODE_W / 2)
    minY = Math.min(minY, n.y)
    maxY = Math.max(maxY, n.y + NODE_H)
  }

  // Shift everything so top-left is at PAD
  const offX = -minX + PAD
  const offY = -minY + PAD
  for (const n of positioned) { n.x += offX; n.y += offY }
  for (const e of layoutEdges) {
    e.from = { x: e.from.x + offX, y: e.from.y + offY }
    e.to = { x: e.to.x + offX, y: e.to.y + offY }
  }

  return {
    positioned,
    layoutEdges,
    fwd,
    bwd,
    width: maxX - minX + PAD * 2,
    height: maxY - minY + PAD * 2,
  }
}

// BFS traversal helper
function reachable(startId, adjMap) {
  const visited = new Set()
  const queue = [startId]
  while (queue.length) {
    const id = queue.shift()
    if (visited.has(id)) continue
    visited.add(id)
    for (const next of adjMap[id] || []) queue.push(next)
  }
  visited.delete(startId)
  return visited
}

function GraphDiagram({ code }) {
  const containerRef = useRef(null)
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 })
  const dragging = useRef(false)
  const lastPos = useRef({ x: 0, y: 0 })
  const [graph, setGraph] = useState(null)
  const [hoverId, setHoverId] = useState(null)

  useEffect(() => {
    const { nodes, edges } = parseMermaidGraph(code)
    const result = layoutGraph(nodes, edges)
    setGraph(result)
    if (containerRef.current && result.width > 0) {
      const contW = containerRef.current.clientWidth
      const contH = containerRef.current.clientHeight
      const fitScale = Math.min(contW / result.width, contH / result.height, 1) * 0.9
      setTransform({ x: 0, y: 0, scale: fitScale })
    }
  }, [code])

  // Compute highlighted sets when hovering
  const { fwdSet, bwdSet } = (() => {
    if (!hoverId || !graph) return { fwdSet: new Set(), bwdSet: new Set() }
    return {
      fwdSet: reachable(hoverId, graph.fwd),  // courses after (dependents) → blue
      bwdSet: reachable(hoverId, graph.bwd),  // courses before (prereqs) → red
    }
  })()

  const onWheel = useCallback((e) => {
    e.preventDefault()
    setTransform((t) => {
      const delta = e.deltaY > 0 ? 0.9 : 1.1
      return { ...t, scale: Math.min(Math.max(t.scale * delta, 0.05), 4) }
    })
  }, [])

  const onMouseDown = useCallback((e) => {
    if (e.button !== 0) return
    dragging.current = true
    lastPos.current = { x: e.clientX, y: e.clientY }
    e.preventDefault()
  }, [])

  const onMouseMove = useCallback((e) => {
    if (!dragging.current) return
    const dx = e.clientX - lastPos.current.x
    const dy = e.clientY - lastPos.current.y
    lastPos.current = { x: e.clientX, y: e.clientY }
    setTransform((t) => ({ ...t, x: t.x + dx, y: t.y + dy }))
  }, [])

  const onMouseUp = useCallback(() => { dragging.current = false }, [])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [onWheel])

  const fitToView = useCallback(() => {
    if (!containerRef.current || !graph) return
    const contW = containerRef.current.clientWidth
    const contH = containerRef.current.clientHeight
    const fitScale = Math.min(contW / graph.width, contH / graph.height, 1) * 0.9
    setTransform({ x: 0, y: 0, scale: fitScale })
  }, [graph])

  if (!graph || graph.positioned.length === 0) return null

  const getNodeColor = (n) => {
    if (!hoverId) return n.color
    if (n.id === hoverId) return '#fff'
    if (bwdSet.has(n.id)) return '#c62828'  // red - before (prerequisites)
    if (fwdSet.has(n.id)) return '#1565c0'  // blue - after (dependents)
    return n.color
  }

  const getNodeOpacity = (n) => {
    if (!hoverId) return 0.9
    if (n.id === hoverId || bwdSet.has(n.id) || fwdSet.has(n.id)) return 1
    return 0.15
  }

  const getEdgeStyle = (e) => {
    if (!hoverId) return { stroke: '#555', opacity: 0.5 }
    const fIs = e.fromId === hoverId
    const tIs = e.toId === hoverId
    const fBwd = bwdSet.has(e.fromId)
    const tBwd = bwdSet.has(e.toId)
    const fFwd = fwdSet.has(e.fromId)
    const tFwd = fwdSet.has(e.toId)
    // Red: both endpoints in prereq chain (bwdSet + hovered)
    if ((fBwd || fIs) && (tBwd || tIs))
      return { stroke: '#c62828', opacity: 0.9 }
    // Blue: both endpoints in dependent chain (fwdSet + hovered)
    if ((fFwd || fIs) && (tFwd || tIs))
      return { stroke: '#1565c0', opacity: 0.9 }
    return { stroke: '#555', opacity: 0.08 }
  }

  return (
    <div className="mermaid-viewer">
      <div className="mermaid-controls">
        <button onClick={() => setTransform((t) => ({ ...t, scale: Math.min(t.scale * 1.2, 4) }))}>+</button>
        <button onClick={() => setTransform((t) => ({ ...t, scale: Math.max(t.scale * 0.8, 0.05) }))}>−</button>
        <button onClick={fitToView}>⟲</button>
      </div>
      <div
        ref={containerRef}
        className="mermaid-container"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => { dragging.current = false; setHoverId(null) }}
      >
        <svg
          className="graph-svg"
          width={graph.width}
          height={graph.height}
          viewBox={`0 0 ${graph.width} ${graph.height}`}
          style={{ transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})` }}
        >
          <defs>
            <marker id="arrow-default" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#666" />
            </marker>
            <marker id="arrow-red" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#c62828" />
            </marker>
            <marker id="arrow-blue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
              <polygon points="0 0, 8 3, 0 6" fill="#1565c0" />
            </marker>
          </defs>
          {/* Edges */}
          {graph.layoutEdges.map((e, i) => {
            const x1 = e.from.x, y1 = e.from.y + NODE_H
            const x2 = e.to.x, y2 = e.to.y
            const midY = (y1 + y2) / 2
            const style = getEdgeStyle(e)
            const markerId = style.stroke === '#c62828' ? 'arrow-red' : style.stroke === '#1565c0' ? 'arrow-blue' : 'arrow-default'
            return (
              <path
                key={`e${i}`}
                d={`M${x1},${y1} C${x1},${midY} ${x2},${midY} ${x2},${y2}`}
                fill="none"
                stroke={style.stroke}
                strokeWidth={style.opacity > 0.5 ? 2.5 : 1.5}
                opacity={style.opacity}
                markerEnd={`url(#${markerId})`}
              />
            )
          })}
          {/* Nodes */}
          {graph.positioned.map((n) => {
            const fill = getNodeColor(n)
            const opacity = getNodeOpacity(n)
            const isHovered = n.id === hoverId
            const textColor = isHovered ? '#000' : '#fff'
            const truncated = n.label.length > 22
            const displayLabel = (!isHovered && truncated) ? n.label.slice(0, 20) + '…' : n.label
            // Expand width on hover if text was truncated
            const w = isHovered && truncated ? Math.max(NODE_W, displayLabel.length * 8 + 24) : NODE_W
            return (
              <g
                key={n.id}
                onMouseEnter={() => setHoverId(n.id)}
                onMouseLeave={() => setHoverId(null)}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={n.x - w / 2}
                  y={n.y}
                  width={w}
                  height={NODE_H}
                  rx={8}
                  ry={8}
                  fill={fill}
                  stroke={isHovered ? '#fff' : fill}
                  strokeWidth={isHovered ? 2.5 : 1.5}
                  opacity={opacity}
                />
                <text
                  x={n.x}
                  y={n.y + NODE_H / 2}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill={textColor}
                  fontSize={11}
                  fontFamily="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
                  opacity={opacity}
                  pointerEvents="none"
                >
                  {displayLabel}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}

const BUCKET_LABELS = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']
const BAR_COLORS = ['#c62828', '#d84315', '#ef6c00', '#f9a825', '#9e9d24', '#689f38', '#2e7d32', '#00838f', '#1565c0', '#283593']

function HistogramChart({ code }) {
  let data
  try { data = JSON.parse(code) } catch { return <pre>{code}</pre> }
  if (!data.entries || data.entries.length === 0) return null

  return (
    <div className="histogram-viewer" dir="rtl">
      <div className="histogram-title">{data.course}</div>
      {data.entries.map((entry, idx) => {
        const maxCount = Math.max(...entry.buckets, 1)
        const BAR_W = 32
        const BAR_MAX_H = 120
        const chartW = entry.buckets.length * (BAR_W + 6) + 60
        const chartH = BAR_MAX_H + 60
        // Average position as fraction of 0-100 range
        const avgFrac = Math.min(Math.max(entry.avg / 100, 0), 1)
        const avgX = 30 + avgFrac * (entry.buckets.length * (BAR_W + 6))

        return (
          <div key={idx} className="histogram-entry">
            <div className="histogram-label">{entry.label}</div>
            <div className="histogram-stats">
              ממוצע: <strong>{entry.avg}</strong> | נבחנים: <strong>{entry.num}</strong>
            </div>
            <svg width={chartW} height={chartH} className="histogram-svg" dir="ltr">
              {/* Bars */}
              {entry.buckets.map((count, i) => {
                const barH = (count / maxCount) * BAR_MAX_H
                const x = 30 + i * (BAR_W + 6)
                const y = BAR_MAX_H - barH + 10
                return (
                  <g key={i}>
                    <rect
                      x={x} y={y}
                      width={BAR_W} height={barH}
                      rx={3} ry={3}
                      fill={BAR_COLORS[i]}
                      opacity={0.85}
                    />
                    {count > 0 && (
                      <text
                        x={x + BAR_W / 2} y={y - 4}
                        textAnchor="middle" fontSize={10} fill="#ccc"
                      >
                        {count}
                      </text>
                    )}
                    <text
                      x={x + BAR_W / 2} y={BAR_MAX_H + 24}
                      textAnchor="middle" fontSize={9} fill="#888"
                    >
                      {BUCKET_LABELS[i]}
                    </text>
                  </g>
                )
              })}
              {/* Average line */}
              <line
                x1={avgX} y1={5} x2={avgX} y2={BAR_MAX_H + 10}
                stroke="#fff" strokeWidth={2} strokeDasharray="4,3" opacity={0.7}
              />
              <text x={avgX} y={BAR_MAX_H + 45} textAnchor="middle" fontSize={10} fill="#fff">
                {entry.avg}
              </text>
            </svg>
          </div>
        )
      })}
    </div>
  )
}

const mdComponents = {
  code({ children, className }) {
    if (className === 'language-mermaid') {
      return <GraphDiagram code={String(children).trim()} />
    }
    if (className === 'language-histogram') {
      return <HistogramChart code={String(children).trim()} />
    }
    return <code className={className}>{children}</code>
  },
  pre({ children }) {
    return <>{children}</>
  },
}

const API_URL = '/v1/chat/completions'

function parseReviews(result) {
  return result.split(/\n\n(?=\[ביקורת \d+)/).filter(Boolean).map((block) => {
    const match = block.match(/^\[(.+?)\]\n([\s\S]*)/)
    if (match) return { meta: match[1], content: match[2].trim() }
    return { meta: '', content: block.trim() }
  })
}

function SourceCard({ output }) {
  const { tool_name, tool_args, tool_result } = output

  if (tool_name === 'search_course_reviews') {
    const reviews = parseReviews(tool_result)
    const filters = [
      tool_args.course_name,
      tool_args.lecturer,
      tool_args.course_type,
    ].filter(Boolean).join(' | ')

    return (
      <div className="source-section">
        <div className="source-section-header">
          {filters || 'חיפוש כללי'} - שאילתה: &quot;{tool_args.query}&quot;
        </div>
        {reviews.map((r, i) => (
          <div key={i} className="source-card">
            {r.meta && <div className="source-meta">{r.meta}</div>}
            <div className="source-content">{r.content}</div>
          </div>
        ))}
      </div>
    )
  }

  if (tool_name === 'kdams_tree') {
    return (
      <div className="source-section">
        <div className="source-section-header">
          עץ קדמים: {tool_args.course_name}
        </div>
        <div className="source-card">
          <pre className="source-pre">{tool_result}</pre>
        </div>
      </div>
    )
  }

  if (tool_name === 'course_grades') {
    // Strip the histogram code block from the raw text for the source card
    const textOnly = tool_result.replace(/```histogram[\s\S]*?```/g, '').trim()
    return (
      <div className="source-section">
        <div className="source-section-header">
          ציונים: {tool_args.course_name}
        </div>
        <div className="source-card">
          <pre className="source-pre">{textOnly}</pre>
        </div>
      </div>
    )
  }

  return null
}

function SourcesDrawer({ toolOutputs }) {
  const [open, setOpen] = useState(false)
  if (!toolOutputs || toolOutputs.length === 0) return null

  return (
    <div className="sources-drawer">
      <button className="sources-drawer-toggle" onClick={() => setOpen((v) => !v)}>
        <span>{open ? '▾' : '◂'}</span>
        <span>נתונים שנמצאו ({toolOutputs.length})</span>
      </button>
      {open && (
        <div className="sources-drawer-body">
          {toolOutputs.map((o, i) => <SourceCard key={i} output={o} />)}
        </div>
      )}
    </div>
  )
}

const INITIAL_SUGGESTIONS = [
  'מה אומרים על מבוא למדעי המחשב?',
  'מי המרצה הכי טוב לאלגוריתמים?',
  'מה הקדמים של מערכות הפעלה?',
  'איזה קורסי בחירה ממליצים?',
]

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'שלום! אני יכול לעזור לך עם שאלות על קורסים, מרצים, קדמים ועוד. שאל אותי משהו.', toolOutputs: [] },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [suggestions, setSuggestions] = useState(INITIAL_SUGGESTIONS)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendText(text) {
    if (!text || loading) return
    setInput('')
    setSuggestions([])
    setLoading(true)

    const userMsg = { role: 'user', content: text, toolOutputs: [] }
    const history = [...messages, userMsg]
    setMessages(history)

    const apiMessages = history
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map(({ role, content }) => ({ role, content }))

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'haifa-rag', messages: apiMessages, stream: false }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      const answer = data.choices[0].message.content
      const toolOutputs = data.tool_outputs || []

      setMessages((prev) => [...prev, { role: 'assistant', content: answer, toolOutputs }])

      if (data.suggestions && data.suggestions.length > 0) {
        setSuggestions(data.suggestions)
      } else {
        setSuggestions([])
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `שגיאה: ${err.message}`, toolOutputs: [] },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function send() {
    sendText(input.trim())
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const isEmbed = new URLSearchParams(window.location.search).get('embed') === '1'

  return (
    <>
      <style>{styles}</style>
      {!isEmbed && (
        <header className="app-header">
          <h1>HAIFA-RAG</h1>
          <span>עוזר לסטודנטים במדעי המחשב | אוניברסיטת חיפה</span>
        </header>
      )}

      <div className="app-main">
        <div className="chat-panel">
          <div className="messages">
            {messages.map((m, i) => (
              <div key={i} className={`msg-row msg-row-${m.role}`}>
                <div className={`msg msg-${m.role}`}>
                  {m.role === 'assistant'
                    ? <div className="md"><Markdown remarkPlugins={[remarkGfm]} components={mdComponents}>{m.content}</Markdown></div>
                    : m.content}
                </div>
                {m.role === 'assistant' && (
                  <SourcesDrawer toolOutputs={m.toolOutputs} />
                )}
              </div>
            ))}
            {loading && (
              <div className="msg-row msg-row-assistant">
                <div className="msg msg-assistant msg-typing">חושב...</div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {suggestions.length > 0 && !loading && (
            <div className="suggestions">
              {suggestions.map((s, i) => (
                <button key={i} className="suggestion-chip" onClick={() => sendText(s)}>
                  {s}
                </button>
              ))}
            </div>
          )}

          <div className="input-area">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="שאל שאלה..."
              rows={1}
              disabled={loading}
            />
            <button onClick={send} disabled={loading || !input.trim()}>
              שלח
            </button>
          </div>
        </div>
      </div>
    </>
  )
}

const styles = `
  :root {
    --bg: #0f0f0f;
    --surface: #1a1a1a;
    --surface2: #242424;
    --border: #333;
    --text: #e0e0e0;
    --text-dim: #888;
    --accent: #6c9eff;
    --user-bg: #1e3a5f;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body, #root {
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  #root {
    display: flex;
    flex-direction: column;
  }

  /* Header */
  .app-header {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  .app-header h1 { font-size: 18px; font-weight: 600; }
  .app-header span { color: var(--text-dim); font-size: 13px; }

  /* Main layout */
  .app-main {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Chat */
  .chat-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  /* Message rows */
  .msg-row {
    display: flex;
    flex-direction: column;
    max-width: 85%;
  }
  .msg-row-user { align-self: flex-start; }
  .msg-row-assistant { align-self: flex-end; }

  .msg {
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.7;
    font-size: 14px;
    word-wrap: break-word;
  }
  .msg-user {
    background: var(--user-bg);
    border-bottom-right-radius: 4px;
    white-space: pre-wrap;
  }
  .msg-assistant {
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg-typing {
    color: var(--text-dim);
    font-style: italic;
  }

  /* Sources drawer (per-message) */
  .sources-drawer {
    margin-top: 4px;
  }
  .sources-drawer-toggle {
    background: none;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--accent);
    padding: 4px 12px;
    font-size: 12px;
    font-family: inherit;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    direction: rtl;
    transition: border-color 0.15s;
  }
  .sources-drawer-toggle:hover {
    border-color: var(--accent);
  }
  .sources-drawer-body {
    margin-top: 6px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface);
    padding: 10px;
    max-height: 400px;
    overflow-y: auto;
  }

  /* Markdown inside messages */
  .md { direction: rtl; }
  .md h1, .md h2, .md h3, .md h4 {
    margin: 12px 0 6px;
    color: var(--text);
  }
  .md h3 { font-size: 15px; }
  .md h4 { font-size: 14px; }
  .md p { margin: 6px 0; }
  .md strong { color: var(--text); }
  .md em { color: var(--accent); font-style: italic; }
  .md ul, .md ol {
    margin: 6px 0;
    padding-right: 20px;
  }
  .md li { margin: 3px 0; }
  .md hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 12px 0;
  }
  .md table {
    border-collapse: collapse;
    margin: 10px 0;
    width: 100%;
    font-size: 13px;
  }
  .md th, .md td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: right;
  }
  .md th {
    background: var(--surface2);
    font-weight: 600;
  }
  .md code {
    background: var(--surface2);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 13px;
  }
  .md pre {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    overflow-x: auto;
    margin: 8px 0;
  }
  .md pre code {
    background: none;
    padding: 0;
  }
  .md blockquote {
    border-right: 3px solid var(--accent);
    padding-right: 10px;
    margin: 8px 0;
    color: var(--text-dim);
  }
  .mermaid-viewer {
    margin: 10px 0;
    direction: ltr;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    background: var(--surface2);
  }
  .mermaid-controls {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 2;
    display: flex;
    gap: 4px;
  }
  .mermaid-controls button {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
  }
  .mermaid-controls button:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  .mermaid-container {
    width: 100%;
    height: 70vh;
    overflow: hidden;
    cursor: grab;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .mermaid-container:active { cursor: grabbing; }
  .graph-svg {
    transform-origin: center center;
    display: block;
  }

  /* Suggestions */
  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 16px;
    border-top: 1px solid var(--border);
    background: var(--surface);
  }
  .suggestion-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    color: var(--accent);
    padding: 6px 14px;
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    direction: rtl;
  }
  .suggestion-chip:hover {
    border-color: var(--accent);
    background: #1e3a5f22;
  }

  /* Input */
  .input-area {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    gap: 8px;
    flex-shrink: 0;
  }
  .input-area textarea {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    padding: 10px 14px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    direction: rtl;
    outline: none;
  }
  .input-area textarea:focus { border-color: var(--accent); }
  .input-area button {
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 10px;
    padding: 0 18px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
  }
  .input-area button:disabled { opacity: 0.4; cursor: default; }

  /* Source cards */
  .source-section { margin-bottom: 12px; }
  .source-section:last-child { margin-bottom: 0; }
  .source-section-header {
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
    direction: rtl;
  }
  .source-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 6px;
    font-size: 13px;
    line-height: 1.5;
  }
  .source-card:last-child { margin-bottom: 0; }
  .source-meta {
    color: var(--accent);
    font-size: 11px;
    margin-bottom: 6px;
    font-weight: 500;
  }
  .source-content {
    color: var(--text-dim);
    direction: rtl;
  }
  .source-pre {
    white-space: pre-wrap;
    font-family: monospace;
    font-size: 12px;
    direction: rtl;
    color: var(--text-dim);
    margin: 0;
  }

  /* Histogram */
  .histogram-viewer {
    margin: 10px 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--surface2);
    padding: 16px;
    overflow-x: auto;
  }
  .histogram-title {
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text);
  }
  .histogram-entry {
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .histogram-entry:last-child {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
  }
  .histogram-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--accent);
    margin-bottom: 4px;
  }
  .histogram-stats {
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 8px;
  }
  .histogram-stats strong {
    color: var(--text);
  }
  .histogram-svg {
    display: block;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
`
