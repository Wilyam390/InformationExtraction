import { LABEL_CFG } from '../data/labels.js'

export default function LabelBadge({ label }) {
  const cfg = LABEL_CFG[label] || { display: label, color: '#888', bg: 'rgba(255,255,255,0.04)', border: 'rgba(255,255,255,0.10)' }
  return (
    <div style={{
      background: cfg.bg,
      border: `1px solid ${cfg.border}`,
      color: cfg.color,
      padding: '12px 16px',
      borderRadius: 'var(--radius-md)',
      display: 'flex', alignItems: 'center', gap: 10,
      fontWeight: 700, fontSize: '1em', letterSpacing: '-0.02em',
    }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: cfg.color,
        boxShadow: `0 0 8px ${cfg.color}`,
        flexShrink: 0, display: 'inline-block',
      }} />
      {cfg.display}
    </div>
  )
}