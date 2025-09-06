import React from 'react';
import { useNavigate } from 'react-router-dom';
import './ResultsPage.css';

const ResultsPage = ({ analysisResult }) => {
  const navigate = useNavigate();

  if (!analysisResult) {
    navigate('/scan');
    return null;
  }

  return (
    <div className="app-page-container">
      <div className="content-card large">
        <button onClick={() => navigate('/scan')} className="back-btn">← Back</button>
        <h2 className="results-title">Scanned Results</h2>
        <h3 className="section-heading">Extracted Text</h3>
        <pre className="extracted-text">{analysisResult.extractedText}</pre>
        <h3 className="section-heading">Detected Warnings</h3>
        <div className="warnings-list">
          {analysisResult.warnings.length > 0 ? (
            analysisResult.warnings.map((warning, i) => (
              <div key={i} className="warning-item">
                <p className="warning-type">⚠ {warning.type}</p>
                <p className="warning-content">{warning.content}</p>
              </div>
            ))
          ) : (
            <p className="no-warnings">No warnings detected.</p>
          )}
          {analysisResult.personalizedWarnings.length > 0 && (
            <div className="warning-item alert">
              <p className="warning-type">🚨 Personalized Allergy Alert</p>
              {analysisResult.personalizedWarnings.map((w, i) => (
                <p key={i} className="warning-content">{w}</p>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsPage;
