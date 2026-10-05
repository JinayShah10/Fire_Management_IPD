import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Play, Terminal, Settings, Map, Calendar,
  Activity, CheckCircle2, XCircle, Clock, Database, FileText,
  Pause, Square
} from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:8000/api';

const REGIONS = [
  "world", "india", "usa", "australia", "brazil", 
  "canada", "indonesia", "russia", "mexico", 
  "argentina", "china", "south_africa", "spain", 
  "portugal", "greece"
];

function App() {
  const [status, setStatus] = useState({ state: 'idle', pid: null });
  const [logs, setLogs] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  
  const [activeTab, setActiveTab] = useState('quick');
  
  // Quick Settings State
  const [region, setRegion] = useState('india');
  const [startDate, setStartDate] = useState('2024-03-01');
  const [endDate, setEndDate] = useState('2024-05-30');
  const [maxImages, setMaxImages] = useState(100);
  
  // Advanced Settings State
  const [configContent, setConfigContent] = useState('');
  const [loading, setLoading] = useState(true);
  
  const [showRawLogs, setShowRawLogs] = useState(false);
  
  const logEndRef = useRef(null);

  // Polling intervals
  useEffect(() => {
    fetchConfig();
    fetchUploadedFiles();
    const statusInterval = setInterval(fetchStatus, 2000);
    const logsInterval = setInterval(fetchLogs, 2000);
    const filesInterval = setInterval(fetchUploadedFiles, 2000);
    
    return () => {
      clearInterval(statusInterval);
      clearInterval(logsInterval);
      clearInterval(filesInterval);
    };
  }, []);

  // Parse basic settings from raw config content on load
  useEffect(() => {
    if (configContent) {
      const rMatch = configContent.match(/region:\s*(\w+)/);
      const sMatch = configContent.match(/start_date:\s*"([^"]+)"/);
      const eMatch = configContent.match(/end_date:\s*"([^"]+)"/);
      const mMatch = configContent.match(/max_images:\s*(\d+)/);
      
      if (rMatch) setRegion(rMatch[1]);
      if (sMatch) setStartDate(sMatch[1]);
      if (eMatch) setEndDate(eMatch[1]);
      if (mMatch) setMaxImages(parseInt(mMatch[1]));
    }
  }, [loading]);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const fetchStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/status`);
      setStatus(res.data);
    } catch (err) {}
  };

  const fetchLogs = async () => {
    try {
      const res = await axios.get(`${API_URL}/logs`);
      setLogs(res.data.logs);
    } catch (err) {}
  };

  const fetchUploadedFiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/uploaded_files`);
      setUploadedFiles(res.data.files || []);
    } catch (err) {}
  };

  const fetchConfig = async () => {
    try {
      const res = await axios.get(`${API_URL}/config_raw`);
      setConfigContent(res.data.content);
      setLoading(false);
    } catch (err) {
      setLoading(false);
    }
  };

  const saveAdvancedConfig = async () => {
    try {
      await axios.post(`${API_URL}/config_raw`, { content: configContent });
      alert("Advanced Configuration saved successfully!");
      fetchConfig();
    } catch (err) {
      alert("Error saving configuration");
    }
  };

  const saveQuickSettings = async () => {
    try {
      await axios.post(`${API_URL}/update_ingestion`, { 
        region: "user_uploaded", 
        start_date: "2000-01-01", 
        end_date: "2100-01-01",
        max_images: maxImages
      });
      // Fetch latest config to reflect changes in advanced tab
      await fetchConfig(); 
    } catch (err) {
      alert("Error saving quick settings");
      throw err;
    }
  };

  const uploadFile = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    let successCount = 0;
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
          await axios.post(`${API_URL}/upload_firms`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          successCount++;
        } catch (err) {
          alert(`Error uploading ${file.name}: ` + (err.response?.data?.detail || err.message));
        }
    }
    
    if (successCount > 0) {
        alert(`Successfully uploaded ${successCount} file(s)!`);
    }
    
    // Clear the input so they can upload the same files again if needed
    e.target.value = null;
  };

  const clearFiles = async () => {
    try {
      await axios.post(`${API_URL}/clear_files`);
      fetchUploadedFiles();
      alert("Successfully cleared uploaded files.");
    } catch (err) {
      alert("Error clearing files");
    }
  };

  const startPipeline = async () => {
    try {
      if (activeTab === 'quick') {
        await saveQuickSettings();
      }
      await axios.post(`${API_URL}/run`);
      fetchStatus();
    } catch (err) {
      if (err.response) {
         alert("Failed to start pipeline: " + err.response.data.detail);
      }
    }
  };

  const resumePipeline = async () => {
    try {
      if (activeTab === 'quick') {
        await saveQuickSettings();
      }
      await axios.post(`${API_URL}/resume`);
      fetchStatus();
    } catch (err) {
      if (err.response) {
         alert("Failed to resume pipeline: " + err.response.data.detail);
      }
    }
  };

  const pausePipeline = async () => {
    try {
      await axios.post(`${API_URL}/pause`);
      fetchStatus();
    } catch (err) {}
  };
  
  const resumePausedPipeline = async () => {
    try {
      await axios.post(`${API_URL}/resume_paused`);
      fetchStatus();
    } catch (err) {}
  };
  
  const stopPipeline = async () => {
    try {
      await axios.post(`${API_URL}/stop`);
      fetchStatus();
    } catch (err) {}
  };

  const StatusIcon = () => {
    switch (status.state) {
      case 'running': return <Activity size={16} />;
      case 'completed': return <CheckCircle2 size={16} />;
      case 'failed': return <XCircle size={16} />;
      default: return <Clock size={16} />;
    }
  };

  const parseProgress = (logText) => {
    const stages = [
      { id: 'fetch', name: 'Data Ingestion (FIRMS)', cmd: 'stage_010' },
      { id: 'filter', name: 'Urban Noise Filtering', cmd: 'stage_015' },
      { id: 'prescreen', name: 'Event Prescreening', cmd: 'stage_020' },
      { id: 'build', name: 'High-Res Imagery Download & Build', cmd: 'stage_030' },
      { id: 'report', name: 'Dataset Evaluation & Reporting', cmd: 'tools/dataset_report' }
    ];

    let currentStageIndex = -1;
    let subProgress = 0;
    let elapsed = "";
    let estimated = "";
    
    const lines = logText.split(/[\n\r]+/);
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      stages.forEach((stage, idx) => {
        if (line.includes(stage.cmd) && line.includes('python3')) {
          currentStageIndex = idx;
          subProgress = 0;
          elapsed = "";
          estimated = "";
        }
      });
      
      const pctMatch = line.match(/(\d+)%/);
      if (pctMatch) {
         subProgress = parseInt(pctMatch[1]);
      }

      const timeMatch = line.match(/\[([0-9:]+)<([^,\]]+)/);
      if (timeMatch) {
         elapsed = timeMatch[1];
         estimated = timeMatch[2];
      }
    }
    
    if (status.state === 'completed') {
       currentStageIndex = stages.length;
    }
    
    return { stages, currentStageIndex, subProgress, elapsed, estimated };
  };

  const { stages, currentStageIndex, subProgress, elapsed, estimated } = parseProgress(logs);

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">
          <Database size={32} color="#3b82f6" />
          Urban Fire Pipeline Engine
        </h1>
        
        <div className={`status-badge ${status.state}`}>
          <StatusIcon />
          <span>{status.state.toUpperCase()}</span>
        </div>
      </header>

      <div className="dashboard-grid">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          <div className="glass-panel">
            <h2 className="panel-header">
              <Settings size={20} />
              Configuration
            </h2>
            
            <div className="tabs">
              <button 
                className={`tab ${activeTab === 'quick' ? 'active' : ''}`}
                onClick={() => setActiveTab('quick')}
              >
                Quick Setup
              </button>
              <button 
                className={`tab ${activeTab === 'advanced' ? 'active' : ''}`}
                onClick={() => setActiveTab('advanced')}
              >
                Advanced (YAML)
              </button>
            </div>

            {loading ? (
              <p>Loading...</p>
            ) : activeTab === 'quick' ? (
              <div className="quick-setup">
                <div className="form-group" style={{ marginBottom: '1.5rem', padding: '1.5rem', border: '2px dashed var(--border-color)', borderRadius: '8px', textAlign: 'center', backgroundColor: 'rgba(255,255,255,0.02)' }}>
                  <label className="form-label" style={{ display: 'block', marginBottom: '0.5rem', fontSize: '1.1rem' }}>
                    <FileText size={20} style={{ display: 'inline', marginRight: '8px', verticalAlign: 'middle', color: '#3b82f6' }} />
                    Upload NASA Bulk CSV Data
                  </label>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: '1.4' }}>
                    1. Go to the <a href="https://firms.modaps.eosdis.nasa.gov/download/" target="_blank" rel="noreferrer" style={{ color: '#3b82f6', textDecoration: 'underline' }}>NASA FIRMS Archive</a><br/>
                    2. Select your region/country and dates, and download the CSV.<br/>
                    3. Upload the CSV file here to completely bypass API rate limits!
                  </p>
                  <input 
                    type="file" 
                    accept=".csv,.zip"
                    multiple
                    onChange={uploadFile}
                    style={{ padding: '0.5rem', backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '4px', border: '1px solid var(--border-color)', width: '100%', maxWidth: '300px', margin: '0 auto', display: 'block', cursor: 'pointer' }}
                  />
                  
                  {uploadedFiles.length > 0 && (
                    <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center', marginBottom: '0.5rem' }}>
                        {uploadedFiles.map((f, i) => (
                          <div key={i} style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', border: '1px solid #10b981', color: '#10b981', padding: '0.25rem 0.75rem', borderRadius: '9999px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <FileText size={12} />
                            {f.name} ({f.size_mb} MB)
                          </div>
                        ))}
                      </div>
                      <button 
                        className="btn secondary" 
                        style={{ padding: '0.3rem 0.6rem', fontSize: '0.8rem', color: '#fca5a5', borderColor: '#fca5a5' }}
                        onClick={clearFiles}
                      >
                        <XCircle size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
                        Clear Uploads
                      </button>
                    </div>
                  )}
                </div>
                
                <div className="form-group" style={{ marginTop: '0.5rem' }}>
                  <label className="form-label">
                    <Database size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
                    Max Images (0 = Process All)
                  </label>
                  <input 
                    type="number" 
                    className="form-input" 
                    value={maxImages}
                    onChange={e => setMaxImages(parseInt(e.target.value) || 0)}
                    min="0"
                  />
                </div>
                
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '1rem', lineHeight: '1.4' }}>
                  The pipeline will automatically read your uploaded local CSV, filter out urban noise, and download matching Sentinel-2 high-res imagery directly.
                </p>
              </div>
            ) : (
              <div className="advanced-setup">
                <textarea 
                  className="config-textarea"
                  value={configContent}
                  onChange={(e) => setConfigContent(e.target.value)}
                  spellCheck="false"
                />
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button className="btn secondary" onClick={saveAdvancedConfig} disabled={status.state === 'running'}>
                    Save YAML
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="glass-panel">
            <h2 className="panel-header">
              <Play size={20} />
              Execution
            </h2>
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              {status.state === 'running' || status.state === 'paused' ? (
                <>
                  <button className="btn" style={{ flex: 2, padding: '1rem' }} onClick={status.state === 'running' ? pausePipeline : resumePausedPipeline}>
                    {status.state === 'running' ? (
                      <><Pause size={20} fill="currentColor" /> Pause</>
                    ) : (
                      <><Play size={20} fill="currentColor" /> Resume (Unpause)</>
                    )}
                  </button>
                  <button className="btn secondary" style={{ flex: 1, padding: '1rem', color: '#fca5a5', borderColor: '#fca5a5' }} onClick={stopPipeline}>
                    <Square size={20} fill="currentColor" /> Stop
                  </button>
                </>
              ) : (
                <>
                  <button 
                    className="btn" 
                    style={{ flex: 2, padding: '1rem' }}
                    onClick={startPipeline}
                    title="Start from the beginning (Fetch -> Filter -> Prescreen -> Build -> Report)"
                  >
                    <Play size={20} fill="currentColor" /> Launch
                  </button>
                  <button 
                    className="btn secondary" 
                    style={{ flex: 1, padding: '1rem' }}
                    onClick={resumePipeline}
                    title="Resume from where it crashed/stopped"
                  >
                    Resume (Fix)
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '1.5rem 1.5rem 1rem 1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2 className="panel-header" style={{ margin: 0 }}>
              <Activity size={20} />
              Pipeline Progress
            </h2>
            <button 
              className="btn secondary" 
              style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
              onClick={() => setShowRawLogs(!showRawLogs)}
            >
              <FileText size={14} />
              {showRawLogs ? "Hide Raw Logs" : "Show Raw Logs"}
            </button>
          </div>
          
          <div style={{ padding: '1.5rem', flex: showRawLogs ? 'none' : '1', overflowY: 'auto' }}>
            {stages.map((stage, idx) => {
              const isCompleted = idx < currentStageIndex || status.state === 'completed';
              const isCurrent = idx === currentStageIndex && status.state === 'running';
              const progress = isCompleted ? 100 : (isCurrent ? subProgress : 0);
              
              return (
                <div key={stage.id} style={{ marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: isCompleted || isCurrent ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                    <span style={{ fontWeight: 500 }}>{stage.name}</span>
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.9rem' }}>
                      {isCurrent && elapsed && (
                         <span style={{ color: 'var(--text-secondary)' }}>
                           Time: {elapsed} {estimated && estimated !== '?' && !estimated.includes('?') ? ` / ETA: ${estimated}` : ''}
                         </span>
                      )}
                      <span>{progress}%</span>
                    </div>
                  </div>
                  <div style={{ height: '8px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ 
                      height: '100%', 
                      width: `${progress}%`, 
                      background: isCompleted ? 'var(--success)' : 'var(--accent)',
                      transition: 'width 0.3s ease',
                      boxShadow: isCurrent ? '0 0 10px var(--accent-glow)' : 'none'
                    }} />
                  </div>
                </div>
              );
            })}
            
            {status.state === 'failed' && (
               <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--error)', borderRadius: '8px', color: '#fca5a5' }}>
                 <XCircle size={20} style={{ display: 'inline', marginRight: '0.5rem', verticalAlign: 'middle' }} />
                 Pipeline failed during {currentStageIndex >= 0 && currentStageIndex < stages.length ? stages[currentStageIndex].name : 'initialization'}. Please check raw logs.
               </div>
            )}
          </div>

          {showRawLogs && (
            <div className="log-viewer" style={{ border: 'none', borderTop: '1px solid var(--border-color)', borderRadius: '0', height: '350px', flex: 'none' }}>
              {logs || "No logs available."}
              <div ref={logEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
