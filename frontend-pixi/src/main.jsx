import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, background: 'red', color: 'white', fontFamily: 'monospace', fontSize: 20, minHeight: '100vh' }}>
          <h1 style={{ fontSize: 60, fontWeight: 900 }}>REACT CRASH</h1>
          <pre style={{ whiteSpace: 'pre-wrap', marginTop: 20 }}>{String(this.state.error)}</pre>
          <pre style={{ whiteSpace: 'pre-wrap', marginTop: 10, fontSize: 14, opacity: 0.7 }}>{this.state.error?.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)
