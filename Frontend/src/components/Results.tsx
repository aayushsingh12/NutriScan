import React from 'react';
import { useLocation } from 'react-router-dom';

interface Flag {
  severity: string;
  category: string;
  description: string;
}

interface IngredientAnalysis {
  name: string;
  classification: string;
  safety_level: string;
  detailed_description: string;
  health_impact: string;
  flags: Flag[];
}

interface AnalysisResult {
  extracted_ingredients: {
    ingredients: string[];
  };
  rag_analysis: {
    ingredient_analyses: IngredientAnalysis[];
    nova_group: number;
    nova_description: string;
    overall_health_assessment: string;
    recommendations: string[];
  };
}

const Results: React.FC = () => {
  const location = useLocation();
  const analysisResult = location.state?.analysisResult as AnalysisResult;

  if (!analysisResult) {
    return <div className="p-4">No analysis results found.</div>;
  }

  return (
    <div className="container mx-auto p-4">
      <h2 className="text-2xl font-bold mb-4">Analysis Results</h2>
      
      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Ingredients List</h3>
        <ul className="list-disc pl-5">
          {analysisResult.extracted_ingredients.ingredients.map((ingredient, index) => (
            <li key={index}>{ingredient}</li>
          ))}
        </ul>
      </div>

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Detailed Analysis</h3>
        {analysisResult.rag_analysis.ingredient_analyses.map((analysis, index) => (
          <div key={index} className="mb-4 p-4 border rounded">
            <h4 className="font-bold">{analysis.name}</h4>
            <p className="mb-2">Safety Level: 
              <span className={`ml-2 px-2 py-1 rounded ${
                analysis.safety_level === 'safe' ? 'bg-green-200' :
                analysis.safety_level === 'caution' ? 'bg-yellow-200' :
                'bg-red-200'
              }`}>
                {analysis.safety_level}
              </span>
            </p>
            <p className="mb-2">{analysis.detailed_description}</p>
            {analysis.flags.length > 0 && (
              <div>
                <p className="font-semibold">Flags:</p>
                <ul className="list-disc pl-5">
                  {analysis.flags.map((flag, flagIndex) => (
                    <li key={flagIndex} className="text-sm">
                      {flag.description} ({flag.severity} severity)
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Overall Assessment</h3>
        <p className="mb-2">NOVA Group: {analysisResult.rag_analysis.nova_group}</p>
        <p className="mb-2">{analysisResult.rag_analysis.nova_description}</p>
        <p className="mb-2">{analysisResult.rag_analysis.overall_health_assessment}</p>
        
        {analysisResult.rag_analysis.recommendations && (
          <div className="mt-4">
            <h4 className="font-semibold">Recommendations:</h4>
            <ul className="list-disc pl-5">
              {analysisResult.rag_analysis.recommendations.map((rec, index) => (
                <li key={index}>{rec}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default Results;
