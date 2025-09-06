import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './AuthPage.css';

const MOCK_USERS = {
  'test@example.com': 'password123',
  'pranav@nutriscan.com': 'hackathonwin',
};

const AuthPage = ({ setUser }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const navigate = useNavigate();

  const handleAuth = (event) => {
    event.preventDefault();
    const email = event.target.email.value;
    const password = event.target.password.value;
    
    setError('');
    setSuccessMessage('');

    if (isLogin) {
      // Login Logic
      if (!MOCK_USERS[email]) {
        setError('No account found with this email. Please sign up.');
      } else if (MOCK_USERS[email] !== password) {
        setError('Wrong password. Please try again.');
      } else {
        setUser({ email });
        navigate('/scan');
      }
    } else {
      // Signup Logic
      if (MOCK_USERS[email]) {
        setError('An account with this email already exists. Please log in.');
      } else {
        // In a real app, you would create a new user here.
        // For this mock, we just proceed.
        MOCK_USERS[email] = password;
        setSuccessMessage('Account created successfully! Please log in.');
        setIsLogin(true); // Redirect to login view
      }
    }
  };

  return (
    <div className="app-page-container">
      <div className="content-card">
        <h2 className="card-title">{isLogin ? 'Login' : 'Sign Up'}</h2>
        <form onSubmit={handleAuth}>
          {error && <div className="auth-error">{error}</div>}
          {successMessage && <div className="auth-success">{successMessage}</div>}
          <input 
            className="auth-input" 
            name="email" 
            type="email" 
            placeholder="Email" 
            required 
          />
          <input 
            className="auth-input" 
            name="password" 
            type="password" 
            placeholder="Password" 
            required 
          />
          <button className="auth-btn" type="submit">
            {isLogin ? 'Login' : 'Sign Up'}
          </button>
        </form>
        <p className="toggle-text">
          {isLogin ? "Don't have an account?" : "Already have an account?"}
          <span onClick={() => { setIsLogin(!isLogin); setError(''); setSuccessMessage(''); }}>
            {isLogin ? ' Sign Up' : ' Login'}
          </span>
        </p>
      </div>
    </div>
  );
};

export default AuthPage;
