import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './ResultsPage.css';

const ResultsPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const analysisResult = location.state?.analysisResult;
  const [shouldRedirect, setShouldRedirect] = React.useState(false);

  React.useEffect(() => {
    if (!analysisResult) {
      setShouldRedirect(true);
    }
  }, [analysisResult]);

  React.useEffect(() => {
    if (shouldRedirect) {
      navigate('/');
    }
  }, [shouldRedirect, navigate]);

  if (shouldRedirect || !analysisResult) {
    return null;
  }

  const { extracted_ingredients, rag_analysis } = analysisResult;

  return (
    <div className="app-page-container">
      <div className="content-card large">
        <button onClick={() => navigate('/')} className="back-btn">← Back</button>
        <h2 className="results-title">Analysis Results</h2>

        {/* Ingredients Section */}
        <div className="section">
          <h3 className="section-heading">Ingredients List</h3>
          <div className="ingredients-list">
            {extracted_ingredients?.ingredients?.map((ingredient, index) => (
              <div key={index} className="ingredient-item">
                {ingredient}
              </div>
            ))}
          </div>
        </div>

        {/* RAG Analysis Section */}
        {rag_analysis && (
          <>
            <div className="section">
              <h3 className="section-heading">Detailed Analysis</h3>
              <div className="analysis-grid">
                {rag_analysis.ingredient_analyses?.map((analysis, index) => (
                  <div key={index} className={`analysis-card ${analysis.safety_level}`}>
                    <h4 className="ingredient-name">{analysis.name}</h4>
                    <div className={`safety-badge ${analysis.safety_level}`}>
                      {analysis.safety_level}
                    </div>
                    <p className="description">{analysis.detailed_description}</p>
                    
                    {analysis.flags && analysis.flags.length > 0 && (
                      <div className="flags-section">
                        <h5>Warnings:</h5>
                        <ul className="flags-list">
                          {analysis.flags.map((flag, i) => (
                            <li key={i} className={`flag-item ${flag.severity}`}>
                              <span className="flag-category">{flag.category}:</span>
                              <span className="flag-description">{flag.description}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Overall Assessment */}
            <div className="section">
              <h3 className="section-heading">Overall Assessment</h3>
              <div className={`nova-group-card nova-${rag_analysis.nova_group}`}>
                <h4>NOVA Group {rag_analysis.nova_group}</h4>
                <p>{rag_analysis.nova_description}</p>
                <p className="health-assessment">{rag_analysis.overall_health_assessment}</p>
              </div>

              {rag_analysis.recommendations && (
                <div className="recommendations">
                  <h4>Recommendations:</h4>
                  <ul>
                    {rag_analysis.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ResultsPage;
