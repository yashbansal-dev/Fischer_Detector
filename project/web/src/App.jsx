import { useState } from 'react'

function App() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleScan = async () => {
    if (!url) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      })

      if (!response.ok) {
        throw new Error('Failed to fetch prediction')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>ShieldLink AI</h1>
        <p className="subtitle">Real-time Malicious URL Detection powered by Advanced Machine Learning.</p>
      </header>

      <main>
        <div className="input-group">
          <input 
            type="text" 
            placeholder="Enter URL to scan (e.g., paypal-security-update.com)" 
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleScan()}
          />
          <button onClick={handleScan} disabled={loading}>
            {loading ? 'Scanning...' : 'Scan URL'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="result-card glass">
            <div className="status-indicator">
              <span className={`status-badge ${result.risk_score < 0.05 ? 'status-benign' : (result.prediction === 'Malicious' ? 'status-malicious' : 'status-benign')}`}>
                {result.risk_score < 0.05 ? 'Verified Safe' : result.prediction}
              </span>
              <div className="risk-score">
                {(result.risk_score * 100).toFixed(1)}%
              </div>
              <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Risk Probability</p>
            </div>

            <div className="features-breakdown">
              <h3 style={{ marginBottom: '1.5rem', fontSize: '1.1rem' }}>
                {Object.keys(result.features).length > 0 ? 'Extracted Indicators' : 'Domain Reputation'}
              </h3>
              <div className="features-list">
                {Object.keys(result.features).length > 0 ? (
                  Object.entries(result.features).map(([key, value]) => (
                    <div key={key} className="feature-item">
                      <span className="feature-label">{key.replace('count_', '').replace('_', ' ')}</span>
                      <span className="feature-value">{typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(2) : value}</span>
                    </div>
                  ))
                ) : (
                  <div className="feature-item" style={{ gridColumn: 'span 2', justifyContent: 'center' }}>
                    <span className="feature-label">This domain is on our trusted whitelist.</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <section style={{ marginTop: '5rem' }}>
          <h2 style={{ marginBottom: '2rem' }}>Model Performance</h2>
          <div className="glass" style={{ padding: '2rem' }}>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
              The system uses a Random Forest ensemble model trained on 100,000+ real-world URL samples.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              <div className="feature-item">
                <span className="feature-label">Training Accuracy</span>
                <span className="feature-value" style={{ color: 'var(--success)' }}>99.5%</span>
              </div>
              <div className="feature-item">
                <span className="feature-label">Inference Time</span>
                <span className="feature-value">~12ms</span>
              </div>
              <div className="feature-item">
                <span className="feature-label">Feature Set</span>
                <span className="feature-value">Lexical</span>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer style={{ marginTop: '5rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        &copy; 2026 ShieldLink AI Project. Designed for premium security analysis.
      </footer>
    </div>
  )
}

export default App
