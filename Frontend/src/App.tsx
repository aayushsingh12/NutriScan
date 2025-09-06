import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ImageUpload from './components/ImageUpload';
import Results from './components/Results';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <div className="container mx-auto py-8">
          <h1 className="text-3xl font-bold text-center mb-8">NutriScan</h1>
          <Routes>
            <Route path="/" element={<ImageUpload />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
