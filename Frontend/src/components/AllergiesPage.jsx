import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./AllergiesPage.css";

const commonAllergies = [
  "Gluten",
  "Dairy",
  "Nuts",
  "Soy",
  "Eggs",
  "Shellfish",
  "Sesame",
];

const AllergiesPage = ({ user }) => {
  const navigate = useNavigate();
  const [selectedAllergies, setSelectedAllergies] = useState([]);
  const [showSuccess, setShowSuccess] = useState(false);

  const isLoggedIn = !!user || localStorage.getItem("isLoggedIn") === "true";

  // Load allergies on mount
  useEffect(() => {
    const stored = localStorage.getItem("user-allergies");
    if (stored) setSelectedAllergies(JSON.parse(stored));
  }, []);

  const handleToggle = (allergy) => {
    if (!isLoggedIn) {
      alert("Please log in to save your allergies.");
      navigate("/auth");
      return;
    }

    setSelectedAllergies((prev) =>
      prev.includes(allergy)
        ? prev.filter((item) => item !== allergy)
        : [...prev, allergy]
    );
  };

  const handleSave = () => {
    if (!isLoggedIn) {
      alert("Please log in to save your allergies.");
      navigate("/auth");
      return;
    }

    localStorage.setItem("user-allergies", JSON.stringify(selectedAllergies));
    setShowSuccess(true);

    // Hide success message after 2 seconds
    setTimeout(() => setShowSuccess(false), 2000);
  };

  return (
    <div className="allergies-page">
      <div className="allergies-card">
        <h1 className="allergies-title">Select Your Allergies</h1>
        <div className="allergies-list">
          {commonAllergies.map((allergy) => (
            <div key={allergy} className="allergy-toggle">
              <span className="allergy-label">{allergy}</span>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={selectedAllergies.includes(allergy)}
                  onChange={() => handleToggle(allergy)}
                />
                <span className="slider"></span>
              </label>
            </div>
          ))}
        </div>

        {/* Save Button */}
        <button className="save-button" onClick={handleSave}>
          Save Allergies
        </button>

        {/* Success Message */}
        {showSuccess && (
          <p className="success-message">✅ Allergies saved successfully!</p>
        )}
      </div>
    </div>
  );
};

export default AllergiesPage;
