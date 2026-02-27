import { useState, useRef, useEffect } from 'react'

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
          {filters || 'חיפוש כללי'} — שאילתה: &quot;{tool_args.query}&quot;
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

  return null
}

function SourcesPanel({ toolOutputs, open, onToggle }) {
  const count = toolOutputs.length

  return (
    <div className={`sources-panel ${open ? '' : 'collapsed'}`}>
      <button className="sources-toggle" onClick={onToggle}>
        {open ? '◀' : '▶'}
      </button>
      {open && (
        <>
          <div className="sources-header">
            <span>נתונים שנמצאו</span>
            {count > 0 && <span className="sources-count">({count})</span>}
          </div>
          <div className="sources-body">
            {count === 0 ? (
              <div className="sources-empty">
                הנתונים שהמערכת מאחזרת יופיעו כאן
              </div>
            ) : (
              toolOutputs.map((o, i) => <SourceCard key={i} output={o} />)
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'שלום! אני יכול לעזור לך עם שאלות על קורסים, מרצים, קדמים ועוד. שאל אותי משהו.' },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [toolOutputs, setToolOutputs] = useState([])
  const [sourcesOpen, setSourcesOpen] = useState(true)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text || loading) return

    const userMsg = { role: 'user', content: text }
    const history = [...messages, userMsg]
    setMessages(history)
    setInput('')
    setLoading(true)

    // Only send user/assistant messages to the API (skip the initial greeting)
    const apiMessages = history
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map(({ role, content }) => ({ role, content }))

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'cs-rag', messages: apiMessages, stream: false }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      const answer = data.choices[0].message.content
      setMessages((prev) => [...prev, { role: 'assistant', content: answer }])

      if (data.tool_outputs && data.tool_outputs.length > 0) {
        setToolOutputs(data.tool_outputs)
        setSourcesOpen(true)
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `שגיאה: ${err.message}` },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <>
      <style>{styles}</style>
      <header className="app-header">
        <h1>CS-RAG</h1>
        <span>עוזר לסטודנטים במדעי המחשב | אוניברסיטת חיפה</span>
      </header>

      <div className="app-main">
        <div className="chat-panel">
          <div className="messages">
            {messages.map((m, i) => (
              <div key={i} className={`msg msg-${m.role}`}>
                {m.content}
              </div>
            ))}
            {loading && (
              <div className="msg msg-assistant msg-typing">חושב...</div>
            )}
            <div ref={bottomRef} />
          </div>

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

        <SourcesPanel
          toolOutputs={toolOutputs}
          open={sourcesOpen}
          onToggle={() => setSourcesOpen((v) => !v)}
        />
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
  .msg {
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.7;
    font-size: 14px;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg-user {
    align-self: flex-start;
    background: var(--user-bg);
    border-bottom-right-radius: 4px;
  }
  .msg-assistant {
    align-self: flex-end;
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg-typing {
    color: var(--text-dim);
    font-style: italic;
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

  /* Sources panel */
  .sources-panel {
    width: 420px;
    border-right: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: width 0.2s;
    position: relative;
    flex-shrink: 0;
  }
  .sources-panel.collapsed { width: 40px; }
  .sources-toggle {
    position: absolute;
    top: 12px;
    left: 8px;
    background: none;
    border: 1px solid var(--border);
    color: var(--text-dim);
    border-radius: 4px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    font-size: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1;
  }
  .sources-toggle:hover { color: var(--accent); border-color: var(--accent); }
  .sources-header {
    padding: 12px 16px 12px 40px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }
  .sources-count { color: var(--text-dim); font-weight: 400; font-size: 12px; }
  .sources-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }
  .sources-empty {
    color: var(--text-dim);
    text-align: center;
    padding: 40px 16px;
    font-size: 13px;
  }

  /* Source cards */
  .source-section { margin-bottom: 16px; }
  .source-section-header {
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  .source-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    line-height: 1.5;
  }
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

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
`
