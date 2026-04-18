export default function ExtractionTable({ fields }) {
  return (
    <div style={{ marginTop: 14 }}>
      <div style={{
        fontSize: '0.68em', fontWeight: 600, letterSpacing: '0.10em',
        textTransform: 'uppercase', color: 'var(--purple)', marginBottom: 8,
      }}>Extracted Invoice Fields</div>
      <table style={{
        width: '100%', borderCollapse: 'collapse',
        border: '1px solid rgba(160,120,255,0.10)',
        borderRadius: 'var(--radius-md)', overflow: 'hidden',
      }}>
        <thead>
          <tr style={{ background: 'rgba(168,85,247,0.05)' }}>
            {['Field', 'Value', 'Status'].map(h => (
              <th key={h} style={{
                padding: '8px 10px',
                textAlign: h === 'Status' ? 'center' : 'left',
                fontSize: '0.68em', fontWeight: 600,
                letterSpacing: '0.07em', textTransform: 'uppercase',
                color: 'var(--text-faint)',
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(fields).map(([k, v]) => (
            <tr key={k} style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}>
              <td style={{ padding: '8px 10px', fontSize: '0.8em', fontWeight: 500, color: 'var(--text)', width: 150 }}>
                {k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </td>
              <td style={{
                padding: '8px 10px', fontSize: '0.8em',
                color: v ? 'var(--text-muted)' : 'var(--text-faint)',
                fontFamily: v ? "'Geist Mono', monospace" : 'inherit',
                fontStyle: v ? 'normal' : 'italic',
              }}>
                {v || 'not found'}
              </td>
              <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                {v
                  ? <span style={{
                      fontSize: '0.68em', fontWeight: 600, letterSpacing: '0.05em',
                      textTransform: 'uppercase', padding: '2px 7px', borderRadius: 5,
                      background: 'rgba(52,211,153,0.10)', color: '#34d399',
                    }}>Found</span>
                  : <span style={{
                      fontSize: '0.68em', fontWeight: 600, letterSpacing: '0.05em',
                      textTransform: 'uppercase', padding: '2px 7px', borderRadius: 5,
                      background: 'rgba(255,255,255,0.04)', color: 'var(--text-faint)',
                    }}>—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}