export default function Card({ children, style = {} }) {
  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      padding: 20,
      boxShadow: '0 4px 20px rgba(0,0,0,0.45)',
      ...style,
    }}>
      {children}
    </div>
  )
}
