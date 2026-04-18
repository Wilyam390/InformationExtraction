import { LABEL_CFG } from '../data/labels.js'

export default function ConfidenceBars({ confidence, predicted }) {
  const sorted = Object.entries(confidence).sort((a, b) => b[1] - a[1])
  const max = sorted[0][1]
  const min = sorted[sorted.length - 1][1]
  const range = Math.max(max - min, 0.01)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9, padding: '4px 0' }}>
      {sorted.map(([lbl, score]) => {
        const pct = Math.round(((score - min) / range) * 100)
        const cfg = LABEL_CFG[lbl] || { display: lbl, color: '#888' }
        const isTop = lbl === predicted
        return (
          <div key={lbl} style={{ display: 'grid', gridTemplateColumns: '140px 1fr 52px', alignItems: 'center', gap: 10 }}>
            <span style={{
              fontSize: '0.78em',
              fontWeight: isTop ? 600 : 400,
              color: isTop ? 'var(--text)' : 'var(--text-faint)',
            }}>
              {cfg.display}
            </span>
            <div style={{ background: 'rgba(255,255,255,0.05)', borderRadius: 99, height: 5, overflow: 'hidden' }}>
              <div style={{
                width: `${pct}%`, height: '100%',
                background: cfg.color, borderRadius: 99,
                transition: 'width 600ms cubic-bezier(0.16,1,0.3,1)',
              }} />
            </div>
            <span style={{
              fontSize: '0.72em', color: 'var(--text-faint)',
              textAlign: 'right', fontFamily: "'Geist Mono', monospace",
            }}>
              {score > 0 ? '+' : ''}{score.toFixed(2)}
            </span>
          </div>
        )
      })}
    </div>
  )
}