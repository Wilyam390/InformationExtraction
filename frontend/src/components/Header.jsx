import Logo from './Logo.jsx'

export default function Header() {
  return (
    <header style={{
      borderBottom: '1px solid var(--border)',
      background: 'rgba(10,10,15,0.85)',
      backdropFilter: 'blur(12px)',
      position: 'sticky', top: 0, zIndex: 50,
    }}>
      <div style={{
        maxWidth: 1080, margin: '0 auto', padding: '0 24px',
        height: 52, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Logo />
          <span style={{ fontWeight: 700, fontSize: '1em', letterSpacing: '-0.03em', color: 'var(--text)' }}>
            Doc<span style={{ color: 'var(--purple)' }}>Classify</span>
          </span>
        </div>
        <span style={{
          fontSize: '0.7em', fontWeight: 600, letterSpacing: '0.08em',
          textTransform: 'uppercase', color: 'var(--text-faint)',
          borderRadius: 99, border: '1px solid var(--border)', padding: '3px 9px',
        }}>
          IE University
        </span>
      </div>
    </header>
  )
}
