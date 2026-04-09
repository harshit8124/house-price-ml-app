import React, { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell
} from "recharts";

function App() {

  const [form, setForm] = useState({
    area_sqft: "",
    bedrooms: "",
    bathrooms: "",
    floors: "",
    garage: "",
    location_score: ""
  });

  const [result, setResult] = useState("");
  const [scores, setScores] = useState([]);

  const colors = ["#00C49F", "#0088FE", "#FFBB28"];

  useEffect(() => {
    fetch("http://127.0.0.1:8000/scores")
      .then(res => res.json())
      .then(data => {
        setScores([
          { name: "Linear", score: data.linear },
          { name: "Tree", score: data.tree },
          { name: "Forest", score: data.forest }
        ]);
      });
  }, []);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const predict = async (model) => {

    // ✅ VALIDATION
    for (let key in form) {
      if (!form[key]) {
        setResult("⚠️ Please fill all fields before predicting");
        return;
      }
    }

    try {
      const res = await fetch(`http://127.0.0.1:8000/predict/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          area_sqft: Number(form.area_sqft),
          bedrooms: Number(form.bedrooms),
          bathrooms: Number(form.bathrooms),
          floors: Number(form.floors),
          garage: Number(form.garage),
          location_score: Number(form.location_score),
        }),
      });

      const data = await res.json();

      // ✅ ERROR HANDLE
      if (data.error) {
        setResult("❌ Error: " + data.error);
      } else {
        setResult(
          `₹ ${data.price} (${data.model.toUpperCase()} | R²: ${data.score})`
        );
      }

    } catch (error) {
      setResult("❌ Server error. Please try again.");
    }
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #1e3c72, #2a5298)",
      color: "white",
      padding: "30px"
    }}>

      {/* TITLE */}
      <h1 style={{
        textAlign: "center",
        fontSize: "32px",
        marginBottom: "30px"
      }}>
        🏠 Smart House Price Predictor
      </h1>

      <div style={{
        display: "flex",
        justifyContent: "center",
        gap: "30px",
        flexWrap: "wrap"
      }}>

        {/* FORM CARD */}
        <div style={{
          background: "rgba(255,255,255,0.1)",
          backdropFilter: "blur(10px)",
          padding: "25px",
          borderRadius: "15px",
          width: "320px",
          boxShadow: "0 8px 20px rgba(0,0,0,0.3)"
        }}>

          <h2 style={{ marginBottom: "15px" }}>Enter Details</h2>

          {Object.keys(form).map((key) => (
            <input
              key={key}
              type="number"
              name={key}
              placeholder={key}
              value={form[key]}
              onChange={handleChange}
              style={{
                width: "100%",
                padding: "10px",
                marginBottom: "10px",
                borderRadius: "8px",
                border: "none",
                outline: "none"
              }}
            />
          ))}

          <div style={{ display: "flex", gap: "10px" }}>
            <button onClick={() => predict("linear")} style={btn("#00C49F")}>
              Linear
            </button>
            <button onClick={() => predict("tree")} style={btn("#0088FE")}>
              Tree
            </button>
            <button onClick={() => predict("forest")} style={btn("#FFBB28")}>
              Forest
            </button>
          </div>

          {/* <h3 style={{ marginTop: "15px" }}>{result}</h3> */}
          <h3 style={{
            marginTop: "15px",
            color: result.includes("⚠️") || result.includes("❌") ? "#ff4d4f" : "white"
          }}>
            {result}
          </h3>
        </div>

        {/* GRAPH CARD */}
        <div style={{
          background: "white",
          padding: "20px",
          borderRadius: "15px",
          color: "black"
        }}>
          <h3>Model Accuracy</h3>

          <BarChart width={400} height={300} data={scores}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="score">
              {scores.map((entry, index) => (
                <Cell key={index} fill={colors[index]} />
              ))}
            </Bar>
          </BarChart>
        </div>

      </div>
    </div>
  );
}

// 🔥 Button Style Function
const btn = (color) => ({
  flex: 1,
  padding: "10px",
  border: "none",
  borderRadius: "8px",
  background: color,
  color: "white",
  cursor: "pointer",
  fontWeight: "bold"
});

export default App;