import React from 'react';

export default function App() {
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      backgroundColor: 'yellow',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      border: '20px solid black',
      fontFamily: 'sans-serif'
    }}>
      <h1 style={{ fontSize: '100px', fontWeight: '900' }}>TEST RENDER</h1>
      <p style={{ fontSize: '30px' }}>If you see this, React is working.</p>
    </div>
  );
}
