import React from 'react';
import './DatabasePage.css';

const DatabasePage = ({ scanDatabase, setScanDatabase }) => {
  // Delete a single item
  const handleDelete = (index) => {
    setScanDatabase((prev) => prev.filter((_, i) => i !== index));
  };

  // Clear all
  const handleClearAll = () => {
    setScanDatabase([]);
  };

  return (
    <div className="database-page-container">
      <div className="database-card">
        {/* Header with title + Clear All */}
        <div className="database-header">
          <h2 className="database-title">Scanned Products Database</h2>
          {scanDatabase.length > 0 && (
            <button className="clear-all-btn" onClick={handleClearAll}>
              Clear All
            </button>
          )}
        </div>

        <p className="database-subtitle">All your scan results stored in one place.</p>

        {scanDatabase.length === 0 ? (
          <p className="no-data-message">No scan results found yet. Start scanning!</p>
        ) : (
          <div className="database-list">
            {scanDatabase.map((item, index) => (
              <div key={index} className="database-item">
                <div className="database-item-header">
                  <h3 className="item-name">{item.productName}</h3>
                  <button
                    className="delete-btn"
                    onClick={() => handleDelete(index)}
                  >
                    ✖
                  </button>
                </div>
                <div className="item-details">
                  <p className="warning-title"><strong>Warnings:</strong></p>
                  {item.warnings.length > 0 ? (
                    <ul className="warnings-list">
                      {item.warnings.map((warning, i) => (
                        <li key={i} className="warning-item">
                          ⚠ <span className="warning-type">{warning.type}</span>: {warning.content}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="no-warnings">No warnings detected ✅</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatabasePage;
