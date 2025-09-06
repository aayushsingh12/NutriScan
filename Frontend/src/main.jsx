import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';

// This is the ONE and ONLY place to import BrowserRouter
import { BrowserRouter } from 'react-router-dom';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* This is the ONE and ONLY place to use BrowserRouter */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
