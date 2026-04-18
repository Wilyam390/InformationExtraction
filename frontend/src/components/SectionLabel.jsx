export default function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: '0.68em', fontWeight: 600, letterSpacing: '0.10em',
      textTransform: 'uppercase', color: 'var(--purple)', marginBottom: 10,
    }}>
      {children}
    </div>
  )
}
