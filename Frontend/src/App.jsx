import React, { useState, useEffect } from "react";
import { Routes, Route, Link, useLocation, useNavigate } from "react-router-dom";
import "./App.css";
import HomePage from "./components/HomePage";
import ScanPage from "./components/ScanPage";
import AuthPage from "./components/AuthPage";
import ResultsPage from "./components/ResultsPage";
import DatabasePage from "./components/DatabasePage";
import AllergiesPage from "./components/AllergiesPage";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isMenuOpen, setIsMenuOpen] = useState(false); // State to toggle hamburger menu

  const handleNavClick = (path) => {
    navigate(path);
    // Close the mobile menu
    setIsMenuOpen(false);
  };

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="navbar">
      <Link to="/" className="nav-logo" onClick={() => handleNavClick("/")}>NutriScan</Link>
      <div className={`nav-links ${isMenuOpen ? "active" : ""}`}>
        <Link to="/scan" className="nav-item" onClick={() => handleNavClick("/scan")}>Scan</Link>
        <Link to="/auth" className="nav-item" onClick={() => handleNavClick("/auth")}>Login/Sign Up</Link>
        <Link to="/database" className="nav-item" onClick={() => handleNavClick("/database")}>Results</Link>
        <Link to="/allergies" className="nav-item" onClick={() => handleNavClick("/allergies")}>Allergies</Link>
      </div>
      <div className="nav-toggle" onClick={toggleMenu}>
        <span></span>
        <span></span>
        <span></span>
      </div>
    </nav>
  );
};

function App() {
  const [user, setUser] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [appMessage, setAppMessage] = useState({ type: "", text: "" });
  const [preferences, setPreferences] = useState({
    gluten: false,
    dairy: false,
    nuts: false,
    other: [],
  });

  // Load data from localStorage on initial render
  const [scanDatabase, setScanDatabase] = useState(() => {
    try {
      const storedData = localStorage.getItem("nutriscan-database");
      return storedData ? JSON.parse(storedData) : [];
    } catch (error) {
      console.error("Failed to parse stored data:", error);
      return [];
    }
  });

  // Save data to localStorage whenever scanDatabase changes
  useEffect(() => {
    try {
      localStorage.setItem("nutriscan-database", JSON.stringify(scanDatabase));
    } catch (error) {
      console.error("Failed to save data to localStorage:", error);
    }
  }, [scanDatabase]);

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem("isLoggedIn");
  };

  return (
    <div className="app-main">
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/auth" element={<AuthPage setUser={setUser} />} />
        <Route
          path="/scan"
          element={
            <ScanPage
              selectedImage={selectedImage}
              setSelectedImage={setSelectedImage}
              analysisResult={analysisResult}
              setAnalysisResult={setAnalysisResult}
              loading={loading}
              setLoading={setLoading}
              appMessage={appMessage}
              setAppMessage={setAppMessage}
              preferences={preferences}
              user={user}
              handleLogout={handleLogout}
              scanDatabase={scanDatabase}
              setScanDatabase={setScanDatabase}
            />
          }
        />
        <Route
          path="/results"
          element={<ResultsPage analysisResult={analysisResult} />}
        />
        <Route
          path="/database"
          element={
            <DatabasePage
              scanDatabase={scanDatabase}
              setScanDatabase={setScanDatabase}
            />
          }
        />
        <Route path="/allergies" element={<AllergiesPage user={user} />} />
      </Routes>
    </div>
  );
}

export default App;