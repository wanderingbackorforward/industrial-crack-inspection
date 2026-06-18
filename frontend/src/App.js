import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('pixel_size_mm', '1.0');
    try {
      const res = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(res.data);
    } catch (err) {
      alert('Inference failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Industrial Crack Inspection</h1>
        <p>Upload a tunnel or industrial surface image to detect and segment cracks.</p>
      </header>
      <main className="container">
        <input type="file" accept="image/*" onChange={handleFileChange} />
        {preview && <img src={preview} alt="preview" className="preview" />}
        <button onClick={handleUpload} disabled={!file || loading}>
          {loading ? 'Analyzing...' : 'Run Inspection'}
        </button>

        {result && (
          <div className="results">
            <h2>Results</h2>
            <p><strong>Cracks detected:</strong> {result.detections.length}</p>
            <p><strong>Crack length:</strong> {result.parameters.length_mm.toFixed(2)} mm</p>
            <p><strong>Mean width:</strong> {result.parameters.mean_width_mm.toFixed(2)} mm</p>
            <p><strong>Max width:</strong> {result.parameters.max_width_mm.toFixed(2)} mm</p>
            <p><strong>Angle:</strong> {result.parameters.angle_deg.toFixed(1)}°</p>
            <p><strong>Fractal dimension:</strong> {result.fractal_dimension.toFixed(4)}</p>
            <details>
              <summary>Raw detections</summary>
              <pre>{JSON.stringify(result.detections, null, 2)}</pre>
            </details>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
