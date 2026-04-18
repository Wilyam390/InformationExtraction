import { useRef, useState } from 'react'
import Card from './Card.jsx'
import SectionLabel from './SectionLabel.jsx'
import { EXAMPLES } from '../data/examples.js'

function ExamplePill({ name, onClick }) {
  const [hov, setHov] = useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        fontSize: '0.74em', fontWeight: 500,
        padding: '4px 12px', borderRadius: 99,
        background: hov ? 'rgba(168,85,247,0.10)' : 'var(--surface-2)',
        border: `1px solid ${hov ? 'rgba(168,85,247,0.40)' : 'var(--border)'}`,
        color: hov ? '#c084fc' : 'var(--text-muted)',
        transition: 'all var(--transition)',
      }}
    >
      {name}
    </button>
  )
}

function FileDropZone({ onFile, fileName }) {
  const ref = useRef()
  const [drag, setDrag] = useState(false)
  return (
    <div
      onClick={() => ref.current.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0]) }}
      style={{
        border: `1px dashed ${drag ? 'rgba(168,85,247,0.55)' : 'rgba(160,120,255,0.22)'}`,
        borderRadius: 'var(--radius-md)',
        padding: '13px 18px',
        background: drag ? 'rgba(168,85,247,0.06)' : 'rgba(168,85,247,0.02)',
        textAlign: 'center', cursor: 'pointer',
        transition: 'all var(--transition)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
      }}
    >
      <input ref={ref} type="file" accept=".txt,.pdf,.png,.jpg,.jpeg,.tiff,.tif"
        style={{ display: 'none' }} onChange={e => onFile(e.target.files[0])} />
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--text-faint)"
        strokeWidth="1.5" strokeLinecap="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <span style={{ fontSize: '0.78em', color: 'var(--text-faint)' }}>
        {fileName
          ? <span style={{ color: 'var(--text-muted)' }}>{fileName}</span>
          : <><span style={{ color: 'var(--purple)' }}>Upload file</span> · txt, pdf, png, jpg, tiff</>}
      </span>
    </div>
  )
}

export default function InputPanel({ text, setText, file, setFile, onClassify, onClear, loading, error }) {
  const [btnHov, setBtnHov] = useState(false)
  const [clrHov, setClrHov] = useState(false)

  return (
    <Card>
      <SectionLabel>Input</SectionLabel>
      <textarea
        value={text}
        onChange={e => { setText(e.target.value); setFile(null) }}
        placeholder="Paste an invoice, email, scientific report, or letter here…"
        rows={13}
        style={{
          width: '100%', resize: 'vertical',
          background: 'var(--surface-2)',
          border: '1px solid rgba(160,120,255,0.12)',
          borderRadius: 'var(--radius-md)',
          padding: '12px 14px',
          color: 'var(--text)',
          fontFamily: "'Geist Mono', monospace",
          fontSize: '0.78em',
          lineHeight: 1.75,
          outline: 'none',
          transition: 'border-color var(--transition), box-shadow var(--transition)',
          marginBottom: 10,
        }}
        onFocus={e => {
          e.target.style.borderColor = 'rgba(124,58,237,0.55)'
          e.target.style.boxShadow = '0 0 0 3px rgba(109,40,217,0.16)'
        }}
        onBlur={e => {
          e.target.style.borderColor = 'rgba(160,120,255,0.12)'
          e.target.style.boxShadow = 'none'
        }}
      />
      <FileDropZone onFile={f => { setFile(f); setText('') }} fileName={file?.name} />
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        <button
          onClick={onClassify}
          disabled={loading}
          onMouseEnter={() => setBtnHov(true)}
          onMouseLeave={() => setBtnHov(false)}
          style={{
            flex: 1,
            background: btnHov ? '#a855f7' : 'var(--purple-deep)',
            color: '#fff', fontWeight: 600, fontSize: '0.84em',
            padding: '9px 0', borderRadius: 'var(--radius-sm)',
            boxShadow: btnHov ? '0 0 20px rgba(168,85,247,0.32)' : 'none',
            transition: 'all var(--transition)',
            opacity: loading ? 0.6 : 1,
          }}
        >
          {loading ? 'Classifying…' : '▶  Classify'}
        </button>
        <button
          onClick={onClear}
          onMouseEnter={() => setClrHov(true)}
          onMouseLeave={() => setClrHov(false)}
          style={{
            background: clrHov ? 'var(--surface-2)' : 'transparent',
            color: clrHov ? 'var(--text)' : 'var(--text-muted)',
            fontSize: '0.84em', fontWeight: 500,
            padding: '9px 18px', borderRadius: 'var(--radius-sm)',
            border: `1px solid ${clrHov ? 'rgba(160,120,255,0.30)' : 'rgba(160,120,255,0.15)'}`,
            transition: 'all var(--transition)',
          }}
        >
          Clear
        </button>
      </div>
      {error && (
        <p style={{ marginTop: 8, fontSize: '0.76em', color: '#f87171', display: 'flex', alignItems: 'center', gap: 5 }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          {error}
        </p>
      )}
      <div style={{ marginTop: 18 }}>
        <div style={{
          fontSize: '0.66em', fontWeight: 600, letterSpacing: '0.09em',
          textTransform: 'uppercase', color: 'var(--text-faint)', marginBottom: 7,
        }}>Quick Examples</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {Object.entries(EXAMPLES).map(([name, txt]) => (
            <ExamplePill key={name} name={name} onClick={() => { setText(txt); setFile(null) }} />
          ))}
        </div>
      </div>
    </Card>
  )
}