import { useState, useCallback } from 'react'
import Header from './components/Header.jsx'
import InputPanel from './components/InputPanel.jsx'
import ResultsPanel from './components/ResultsPanel.jsx'
import Footer from './components/Footer.jsx'

export default function App() {
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const classify = useCallback(async () => {
    if (!text.trim() && !file) { setError('Please paste text or upload a file.'); return }
    setError(null); setLoading(true); setResult(null)
    try {
      let res
      if (file) {
        const fd = new FormData()
        fd.append('file', file)
        res = await fetch('/api/classify-file', { method: 'POST', body: fd })
      } else {
        res = await fetch('/api/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        })
      }
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Classification failed.') }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [text, file])

  const clear = () => { setText(''); setFile(null); setResult(null); setError(null) }

  return (
    <>
      <Header />
      <main style={{ flex:1, maxWidth:1080, margin:'0 auto', width:'100%', padding:'28px 24px 48px' }}>
        <p style={{ fontSize:'0.75em', color:'var(--text-faint)', marginBottom:24, letterSpacing:'0.02em' }}>
          Document Classification &amp; Invoice Extraction · AI: Statistical Learning and Prediction
        </p>
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, alignItems:'start' }}>
          <InputPanel text={text} setText={setText} file={file} setFile={setFile}
            onClassify={classify} onClear={clear} loading={loading} error={error} />
          <ResultsPanel result={result} loading={loading} />
        </div>
      </main>
      <Footer />
    </>
  )
}
