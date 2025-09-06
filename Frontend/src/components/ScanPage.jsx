import React from 'react';
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
    setAppMessage({ type: 'info', text: 'Uploading image to backend...' });

    try {
      const response = await fetch(selectedImage);
      const blob = await response.blob();

      const formData = new FormData();
      formData.append("file", blob, "image.jpg");

      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Server error: ${res.status}`);

      const data = await res.json();

      setAnalysisResult(data);
      setScanDatabase(prevDatabase => [...prevDatabase, data]);

      navigate('/results');
      setAppMessage({ type: 'success', text: 'Scan complete!' });

    } catch (error) {
      console.error(error);
      setAppMessage({ type: 'error', text: 'Error during scan. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  // 🟢 Return JSX
  return (
    <div>
      {analysisResult?.rag_analysis ? (
        <>
          <h3>Ingredients</h3>
          <ul>
            {analysisResult.extracted_ingredients.ingredients.map((ing, idx) => (
              <li key={idx}>{ing}</li>
            ))}
          </ul>

          <h3>RAG Analysis</h3>
          <pre>{JSON.stringify(analysisResult.rag_analysis, null, 2)}</pre>
        </>
      ) : (
        <>
          <input type="file" accept="image/*" onChange={handleImageUpload} />
          <button onClick={handleScanNow} disabled={loading}>
            {loading ? "Scanning..." : "Scan Now"}
          </button>
          {appMessage?.text && <p className={appMessage.type}>{appMessage.text}</p>}
        </>
      )}
    </div>
  );
};

export default ScanPage;
