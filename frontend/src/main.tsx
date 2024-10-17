import App from "../../src/App.tsx";
import "./index.css";

import React from "react";
import ReactDOM from "react-dom"; // Use the main react-dom import

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById("root") // This targets your root element
);
