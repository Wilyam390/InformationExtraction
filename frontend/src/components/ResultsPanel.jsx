import Card from './Card.jsx'
import SectionLabel from './SectionLabel.jsx'
import LabelBadge from './LabelBadge.jsx'
import ConfidenceBars from './ConfidenceBars.jsx'
import ExtractionTable from './ExtractionTable.jsx'

function Spinner() {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', gap: 12, padding: '40px 0', color: 'var(--text-faint)',
    }}>
      <div style={{
        width: 28, height: 28,
        border: '2px solid rgba(168,85,247,0.2)',
        borderTopColor: 'var(--purple)',
        borderRadius: '50%',
        animation: 'spin 700ms linear infinite',
      }} />
      <span style={{ fontSize: '0.78em', letterSpacing: '0.04em' }}>Classifying…</span>
    </div>
  )
}

function EmptyState() {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      textAlign: 'center', padding: '36px 16px', color: 'var(--text-faint)',
    }}>
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="1.2" strokeLinecap="round" style={{ marginBottom: 12, opacity: 0.45 }}>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
      </svg>
      <p style={{ fontSize: '0.82em', marginBottom: 4, color: 'var(--text-muted)' }}>No results yet</p>
      <p style={{ fontSize: '0.74em', maxWidth: '26ch' }}>Paste a document or load an example, then click Classify</p>
    </div>
  )
}

export default function ResultsPanel({ result, loading }) {
  return (
    <Card>
      <SectionLabel>Results</SectionLabel>
      {loading ? <Spinner /> : result ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14, animation: 'fadeIn 300ms ease' }}>
          <div>
            <div style={{
              fontSize: '0.66em', fontWeight: 600, letterSpacing: '0.08em',
              textTransform: 'uppercase', color: 'var(--text-faint)', marginBottom: 6,
            }}>
              Predicted Category
              {result.ocr_used && <span style={{ marginLeft: 8, color: 'var(--purple)' }}>· OCR</span>}
            </div>
            <LabelBadge label={result.label} />
          </div>
          <div>
            <div style={{
              fontSize: '0.66em', fontWeight: 600, letterSpacing: '0.08em',
              textTransform: 'uppercase', color: 'var(--text-faint)', marginBottom: 8,
            }}>Confidence Scores</div>
            <ConfidenceBars confidence={result.confidence} predicted={result.label} />
          </div>
          {result.extraction && <ExtractionTable fields={result.extraction} />}
        </div>
      ) : <EmptyState />}
    </Card>
  )
}