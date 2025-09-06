import React from 'react';
import Tesseract from 'tesseract.js';
import { useNavigate } from 'react-router-dom';
import './ScanPage.css';

const ScanPage = ({
  selectedImage,
  setSelectedImage,
  analysisResult,
  setAnalysisResult,
  loading,
  setLoading,
  appMessage,
  setAppMessage,
  preferences,
  scanDatabase,
  setScanDatabase
}) => {
  const navigate = useNavigate();

  const analyzeTextForWarnings = (text) => {
    const warnings = [];
    const lowerText = text.toLowerCase();
    if (lowerText.includes('sugar')) warnings.push({ type: 'High Sugar', content: 'Contains added sugar' });
    if (lowerText.includes('salt') || lowerText.includes('sodium')) warnings.push({ type: 'High Sodium', content: 'May have high salt content' });
    if (lowerText.includes('gluten')) warnings.push({ type: 'Gluten', content: 'Contains gluten' });
    if (lowerText.includes('milk') || lowerText.includes('dairy')) warnings.push({ type: 'Dairy', content: 'Contains milk or dairy products' });
    if (lowerText.includes('preservative')) warnings.push({ type: 'Preservatives', content: 'Contains preservatives' });
    return warnings;
  };

  const handleImageUpload = (event) => {
    console.log('handleImageUpload triggered'); // Debug log
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result);
        setAppMessage({ type: '', text: '' });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleScanNow = async () => {
    console.log('handleScanNow triggered'); // Debug log
    if (!selectedImage) {
      setAppMessage({ type: 'error', text: 'Please select an image to scan.' });
      return;
    }
    setLoading(true);
    setAnalysisResult(null);
    setAppMessage({ type: 'info', text: 'Running OCR scan...' });

    try {
      const result = await Tesseract.recognize(selectedImage, 'eng');
      const extractedText = result.data.text;
      const warnings = analyzeTextForWarnings(extractedText);

      const finalResult = {
        productName: 'Scanned Product',
        extractedText,
        warnings,
        personalizedWarnings: [],
      };

      if (preferences.nuts) finalResult.personalizedWarnings.push('⚠ May contain traces of nuts.');
      if (preferences.gluten) finalResult.personalizedWarnings.push('⚠ Product may not be gluten-free.');
      if (preferences.dairy) finalResult.personalizedWarnings.push('⚠ Contains milk derivatives.');

      setAnalysisResult(finalResult);
      setScanDatabase(prevDatabase => [...prevDatabase, finalResult]);

      navigate('/results');
      setAppMessage({ type: 'success', text: 'Scan complete!' });
    } catch (error) {
      console.error(error);
      setAppMessage({ type: 'error', text: 'Error during scan. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-page-container">
      <div className="scan-card-horizontal">
        {/* Left Section - Image */}
        <div className="scan-image-container">
          {selectedImage ? (
            <div className="image-wrapper">
              <img src={selectedImage} alt="Label Preview" className="scan-image" />
              <button 
                className="close-btn" 
                onClick={() => {
                  setSelectedImage(null);
                  setAppMessage({ type: '', text: '' });
                }}
              >✖</button>
            </div>
          ) : (
            <div className="placeholder">
              <p>No image uploaded yet</p>
              <p className="placeholder-sub">Upload or capture a label to begin.</p>
            </div>
          )}
        </div>

        {/* Right Section - Info and Controls */}
        <div className="scan-right-panel">
          <p className="scan-tagline">
            Scan a label to find out what's really in your food.
          </p>

          <div className="scan-controls">
            <label htmlFor="file-upload" className="upload-btn">
              Upload Image
              <input
                id="file-upload"
                type="file"
                onChange={handleImageUpload}
                className="file-input"
                accept="image/*"
                capture="environment"
                style={{ position: 'absolute', opacity: 0, width: '100%', height: '100%', top: 0, left: 0, cursor: 'pointer' }}
              />
            </label>

            <button
              onClick={handleScanNow}
              className="scan-now-btn"
              disabled={loading || !selectedImage}
            >
              {loading ? 'Scanning...' : 'Scan Now'}
            </button>
          </div>

          {appMessage.text && (
            <div className={`message ${appMessage.type}`}>
              {appMessage.text}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ScanPage;