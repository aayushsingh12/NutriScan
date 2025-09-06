import React from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

const HomePage = () => {
  const navigate = useNavigate();

  const handleButtonClick = () => {
    navigate('/scan');
  };

  return (
    <div className="app-page-container">
      <div className="homepage-hero">
        <h1 className="homepage-title">What's in your food today?</h1>
        <p className="homepage-tagline">
          Scan a label to find out what's really inside.
        </p>
        
      </div>
    </div>
  );
};



export default HomePage;
