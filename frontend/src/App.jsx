
import WaterCharts from "./WaterCharts";
import Insights from "./Insights";
import { useEffect, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:5000";

const modelResults = [
  {
    name: "Original Random Forest",
    frequency: "Daily",
    mae: 1.437,
    rmse: 2.144,
    description: "Uses historical reservoir storage patterns and calendar features.",
    best: true,
  },
  {
    name: "Climate Random Forest",
    frequency: "Daily",
    mae: 1.451,
    rmse: 2.161,
    description: "Uses storage history along with rainfall and evaporation features.",
  },
  {
    name: "SARIMA",
    frequency: "Monthly",
    mae: 15.029,
    rmse: 16.582,
    description: "Time-series model that captures trend and seasonal patterns.",
  },
  {
    name: "SARIMAX",
    frequency: "Monthly",
    mae: 14.175,
    rmse: 15.947,
    description: "Seasonal time-series model with additional external variables.",
  },
];

function ModelPerformance() {
  const dailyModels = modelResults.filter((model) => model.frequency === "Daily");
  const monthlyModels = modelResults.filter((model) => model.frequency === "Monthly");

  return (
    <div className="performance-page">
      <section className="performance-intro">
        <div>
          <p className="eyebrow">MODEL EVALUATION</p>
          <h2>Forecasting Model Comparison</h2>
          <p>
            Lower MAE and RMSE indicate smaller prediction errors. Daily and
            monthly models are evaluated separately because they use different
            time intervals.
          </p>
        </div>
        <span className="panel-tag">4 Models Evaluated</span>
      </section>

      <section className="performance-group">
        <div className="performance-group-heading">
          <h2>Daily Forecasting Models</h2>
          <span>Errors are in percentage points</span>
        </div>
        <div className="performance-grid">
          {dailyModels.map((model) => (
            <article className={`performance-card ${model.best ? "best-model" : ""}`} key={model.name}>
              {model.best && <span className="best-label">Best daily result</span>}
              <div className="performance-card-top">
                <h3>{model.name}</h3>
                <span className="frequency-tag">{model.frequency}</span>
              </div>
              <p className="performance-description">{model.description}</p>
              <div className="performance-metrics">
                <div><span>MAE</span><strong>{model.mae.toFixed(3)}</strong></div>
                <div><span>RMSE</span><strong>{model.rmse.toFixed(3)}</strong></div>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="performance-group">
        <div className="performance-group-heading">
          <h2>Monthly Time-Series Models</h2>
          <span>Errors are in percentage points</span>
        </div>
        <div className="performance-grid">
          {monthlyModels.map((model) => (
            <article className="performance-card" key={model.name}>
              <div className="performance-card-top">
                <h3>{model.name}</h3>
                <span className="frequency-tag">{model.frequency}</span>
              </div>
              <p className="performance-description">{model.description}</p>
              <div className="performance-metrics">
                <div><span>MAE</span><strong>{model.mae.toFixed(3)}</strong></div>
                <div><span>RMSE</span><strong>{model.rmse.toFixed(3)}</strong></div>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="performance-notes">
        <h2>How to read these metrics</h2>
        <ul>
          <li><strong>MAE (Mean Absolute Error):</strong> average absolute difference between actual and predicted water availability.</li>
          <li><strong>RMSE (Root Mean Squared Error):</strong> gives greater weight to larger prediction errors.</li>
          <li><strong>Lower values are better</strong> when models are evaluated on the same dataset and time interval.</li>
        </ul>
        <p className="comparison-warning">
          Note: Daily and monthly results should not be compared directly. The
          models use different evaluation frequencies and test periods.
        </p>
        <p className="comparison-warning">
          The original Random Forest has the lowest reported error among the
          daily models. SARIMAX performs better than SARIMA among the monthly
          models. The monthly models did not outperform the monthly persistence
          baseline in the reported evaluation.
        </p>
      </section>
    </div>
  );
}

function ScenarioSimulator({ currentPercent }) {
  const [rainfall, setRainfall] = useState("0.5");
  const [grossEvap, setGrossEvap] = useState("2.04");
  const [currentFullness, setCurrentFullness] = useState(
    currentPercent === null ? "75" : String(Number(currentPercent.toFixed(2)))
  );
  const [result, setResult] = useState(null);
  const [simulatorError, setSimulatorError] = useState("");
  const [simulating, setSimulating] = useState(false);

  useEffect(() => {
    if (currentPercent !== null && result === null) {
      setCurrentFullness(String(Number(currentPercent.toFixed(2))));
    }
  }, [currentPercent, result]);

  async function runSimulation(event) {
    event.preventDefault();
    setSimulatorError("");
    setResult(null);

    const rain = Number(rainfall);
    const evaporation = Number(grossEvap);
    const fullness = Number(currentFullness);

    if ([rain, evaporation, fullness].some((value) => !Number.isFinite(value))) {
      setSimulatorError("Please enter valid numbers in all fields.");
      return;
    }
    if (rain < 0 || rain > 20 || evaporation < 0 || evaporation > 20) {
      setSimulatorError("Rainfall and gross evaporation must be between 0 and 20 inches.");
      return;
    }
    if (fullness < 0 || fullness > 100) {
      setSimulatorError("Current fullness must be between 0 and 100%.");
      return;
    }

    setSimulating(true);
    try {
      const response = await fetch(`${API}/api/scenario-predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          rainfall: rain,
          gross_evap: evaporation,
          current_fullness: fullness,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || data.message || "Scenario prediction failed.");
      }
      const predicted = Number(
        data.predicted_fullness ?? data.predicted_percent_full ?? data.prediction
      );
      if (!Number.isFinite(predicted)) {
        throw new Error("The backend response did not contain a valid predicted fullness value.");
      }
      setResult({
        predicted,
        date: data.prediction_date ?? data.date ?? "Next day",
        model: data.model ?? "Climate Random Forest",
        netEvap: data.net_evap ?? evaporation - rain,
      });
    } catch (err) {
      setSimulatorError(`${err.message} Check that your Flask backend and scenario API are running.`);
    } finally {
      setSimulating(false);
    }
  }

  return (
    <div className="scenario-page">
      <section className="scenario-intro">
        <p className="eyebrow">WHAT-IF ANALYSIS</p>
        <h2>Rainfall-Based Scenario Simulator</h2>
        <p>
          Enter rainfall, evaporation and the current reservoir fullness to estimate
          the next-day water availability using the climate-aware model.
        </p>
      </section>

      <section className="scenario-layout">
        <form className="panel scenario-form" onSubmit={runSimulation}>
          <h2>Scenario inputs</h2>
          <p className="scenario-help">Enter climate values in inches and fullness as a percentage.</p>

          <label htmlFor="scenario-rainfall">Rainfall (inches)</label>
          <input id="scenario-rainfall" type="number" min="0" max="20" step="0.01" value={rainfall} onChange={(e) => setRainfall(e.target.value)} required />

          <label htmlFor="scenario-evaporation">Gross evaporation (inches)</label>
          <input id="scenario-evaporation" type="number" min="0" max="20" step="0.01" value={grossEvap} onChange={(e) => setGrossEvap(e.target.value)} required />

          <label htmlFor="scenario-fullness">Current reservoir fullness (%)</label>
          <input id="scenario-fullness" type="number" min="0" max="100" step="0.01" value={currentFullness} onChange={(e) => setCurrentFullness(e.target.value)} required />

          <p className="scenario-net-evap">Net evaporation: <strong>{(Number(grossEvap) - Number(rainfall)).toFixed(2)} inches</strong> (gross evaporation − rainfall)</p>
          <button className="scenario-submit" type="submit" disabled={simulating}>
            {simulating ? "Predicting..." : "Run Scenario Prediction"}
          </button>
          {simulatorError && <p className="scenario-error" role="alert">{simulatorError}</p>}
        </form>

        <section className="panel scenario-result" aria-live="polite">
          <h2>Prediction result</h2>
          {!result ? (
            <div className="scenario-empty">
              <span>💧</span>
              <p>Your predicted reservoir fullness will appear here after you run a scenario.</p>
            </div>
          ) : (
            <>
              <p className="scenario-result-date">Prediction for {String(result.date).slice(0, 10)}</p>
              <div className="scenario-predicted-value">{result.predicted.toFixed(2)}%</div>
              <p className="scenario-result-label">Predicted reservoir fullness</p>
              <div className="scenario-comparison">
                <div><span>Current fullness</span><strong>{Number(currentFullness).toFixed(2)}%</strong></div>
                <div><span>Estimated change</span><strong className={result.predicted - Number(currentFullness) >= 0 ? "scenario-positive" : "scenario-negative"}>{result.predicted - Number(currentFullness) >= 0 ? "+" : ""}{(result.predicted - Number(currentFullness)).toFixed(2)} pp</strong></div>
                <div><span>Net evaporation</span><strong>{Number(result.netEvap).toFixed(2)} in</strong></div>
              </div>
              <p className="scenario-model-label">Model: {result.model}</p>
            </>
          )}
        </section>
      </section>

      <p className="scenario-disclaimer">
        Experimental estimate only. The climate inputs represent the monthly climate measurements used by the training dataset; they should not be interpreted as validated daily rainfall totals. This prediction is not an official reservoir forecast.
      </p>
    </div>
  );
}

function App() {
  const [history, setHistory] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");

  useEffect(() => {
    async function loadDashboard() {
      try {
        const responses = await Promise.all([
          fetch(`${API}/api/history?limit=34073`),
          fetch(`${API}/api/forecast?days=7`),
          fetch(`${API}/api/metrics`),
          fetch(`${API}/api/alerts`),
        ]);

        if (responses.some((response) => !response.ok)) {
          throw new Error("One or more backend APIs returned an error.");
        }

        const [historyData, forecastData, metricsData, alertsData] =
          await Promise.all(
            responses.map((response) => response.json())
          );

        setHistory(
          Array.isArray(historyData)
            ? historyData
            : historyData.history ?? historyData.data ?? []
        );

        setForecast(
          Array.isArray(forecastData)
            ? forecastData
            : forecastData.forecast ?? forecastData.predictions ?? []
        );

        setMetrics(metricsData);

        setAlerts(
          Array.isArray(alertsData)
            ? alertsData
            : alertsData.alerts ?? []
        );
      } catch (err) {
        setError(
          "Could not load dashboard data. Check that your Flask backend is running."
        );
        console.error(err);
      } finally {
        setLoading(false);
      }
    }

    loadDashboard();
  }, []);

  const latest = history.length
    ? history[history.length - 1]
    : null;

  function getPercent(item) {
    if (!item) return null;

    const value =
      item.percent_full ??
      item.predicted_percent_full ??
      item.prediction ??
      item.predicted ??
      item.percentage;

    return value !== undefined &&
      value !== null &&
      Number.isFinite(Number(value))
      ? Number(value)
      : null;
  }

  const currentPercent = getPercent(latest);

  return (
    <div className="app">
      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">WATER RESOURCE MONITORING</p>
            <h1>
              {activeTab === "overview"
                ? "Dashboard Overview"
                : activeTab === "sustainability"
                  ? "Water Sustainability Score"
                  : activeTab === "trends"
                    ? "Seasonal and Long-Term Trend Analysis"
                    : activeTab === "scarcity"
                      ? "Water Scarcity Risk Assessment"
                      : activeTab === "performance"
                        ? "Model Performance"
                        : "Scenario Simulator"}
            </h1>

            <p className="subtitle">
              {activeTab === "overview"
                ? "Monitor historical water availability and explore future predictions."
                : activeTab === "sustainability"
                  ? "Understand recent reservoir availability through an indicative score."
                  : activeTab === "trends"
                    ? "Explore monthly patterns and changes in recorded water availability."
                    : activeTab === "scarcity"
                      ? "Review current availability and forecast-based risk indicators."
                      : activeTab === "performance"
                        ? "Compare the accuracy of daily and monthly forecasting models."
                        : "Explore how rainfall and evaporation scenarios may affect water availability."}
            </p>
          </div>

          <div className="live-badge">
            <span className="status-dot"></span>
            Backend connected
          </div>
        </header>

        {/* Horizontal feature tabs */}
        <nav className="feature-tabs" aria-label="Dashboard features">
          <button
            className={
              activeTab === "overview"
                ? "feature-tab active"
                : "feature-tab"
            }
            onClick={() => setActiveTab("overview")}
          >
            Overview
          </button>

          <button
            className={
              activeTab === "sustainability"
                ? "feature-tab active"
                : "feature-tab"
            }
            onClick={() => setActiveTab("sustainability")}
          >
            Sustainability Score
          </button>

          <button
            className={
              activeTab === "trends"
                ? "feature-tab active"
                : "feature-tab"
            }
            onClick={() => setActiveTab("trends")}
          >
            Trend Analysis
          </button>

          <button
            className={
              activeTab === "scarcity"
                ? "feature-tab active"
                : "feature-tab"
            }
            onClick={() => setActiveTab("scarcity")}
          >
            Scarcity Risk
          </button>

          <button
            className={
              activeTab === "performance"
                ? "feature-tab active"
                : "feature-tab"
            }
            onClick={() => setActiveTab("performance")}
          >
            Model Performance
          </button>

          <button
            className={activeTab === "simulator" ? "feature-tab active" : "feature-tab"}
            onClick={() => setActiveTab("simulator")}
          >
            Scenario Simulator
          </button>
        </nav>

        {loading && (
          <div className="message">Loading AquaPredict data...</div>
        )}

        {error && (
          <div className="error-message">{error}</div>
        )}

        {!loading && !error && (
          <>
            {activeTab === "overview" ? (
              <>
                {/* Dashboard summary cards */}
                <section className="stats-grid" id="overview">
                  <article className="stat-card">
                    <div className="stat-top">
                      <span>Latest Storage Level</span>
                      <span className="stat-icon blue">💧</span>
                    </div>

                    <h2>
                      {currentPercent === null
                        ? "—"
                        : `${currentPercent.toFixed(2)}%`}
                    </h2>

                    <p>Latest recorded percentage full</p>
                  </article>

                  <article className="stat-card">
                    <div className="stat-top">
                      <span>Forecast Horizon</span>
                      <span className="stat-icon purple">📈</span>
                    </div>

                    <h2>{forecast.length || 7} Days</h2>
                    <p>Future water availability predictions</p>
                  </article>

                  <article className="stat-card">
                    <div className="stat-top">
                      <span>Historical Records</span>
                      <span className="stat-icon green">▤</span>
                    </div>

                    <h2>{history.length.toLocaleString()}</h2>
                    <p>Records received from the backend</p>
                  </article>

                  <article className="stat-card">
                    <div className="stat-top">
                      <span>Active Alerts</span>
                      <span className="stat-icon orange">⚠</span>
                    </div>

                    <h2>{alerts.length}</h2>
                    <p>Alerts returned by the backend</p>
                  </article>
                </section>

                {/* Existing interactive charts */}
                <WaterCharts
                  history={history}
                  forecast={forecast}
                />

                <section className="content-grid">
                  {/* Forecast panel */}
                  <article className="panel" id="forecast">
                    <div className="panel-heading">
                      <div>
                        <h2>7-Day Water Availability Forecast</h2>
                        <p>
                          Predictions generated by your machine
                          learning model
                        </p>
                      </div>

                      <span className="panel-tag">ML Forecast</span>
                    </div>

                    {forecast.length === 0 ? (
                      <p className="empty-state">
                        No forecast records returned.
                      </p>
                    ) : (
                      <div className="forecast-list">
                        {forecast.map((item, index) => {
                          const value = getPercent(item);

                          const date =
                            item.date ??
                            item.forecast_date ??
                            item.prediction_date ??
                            `Day ${index + 1}`;

                          return (
                            <div
                              className="forecast-row"
                              key={`${date}-${index}`}
                            >
                              <span className="forecast-date">
                                {String(date).slice(0, 10)}
                              </span>

                              <div className="progress-track">
                                <div
                                  className="progress-fill"
                                  style={{
                                    width: `${Math.max(
                                      0,
                                      Math.min(100, value ?? 0)
                                    )}%`,
                                  }}
                                />
                              </div>

                              <strong>
                                {value === null
                                  ? "—"
                                  : `${value.toFixed(2)}%`}
                              </strong>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    <p className="chart-note">
                      Percentages are model estimates, not guaranteed
                      future conditions.
                    </p>
                  </article>

                  {/* Alerts panel */}
                  <article className="panel" id="alerts">
                    <div className="panel-heading">
                      <div>
                        <h2>Water Alerts</h2>
                        <p>Messages returned by your alert API</p>
                      </div>
                    </div>

                    {alerts.length === 0 ? (
                      <div className="empty-state">
                        <span className="empty-icon">✓</span>
                        <h3>No alerts returned</h3>
                        <p>
                          This means the API returned an empty
                          alert list.
                        </p>
                      </div>
                    ) : (
                      <div className="alerts-list">
                        {alerts.map((alert, index) => (
                          <div
                            className="alert-item"
                            key={index}
                          >
                            <span className="alert-symbol">⚠</span>

                            <div>
                              <h3>
                                {alert.title ??
                                  alert.type ??
                                  "Water status alert"}
                              </h3>

                              <p>
                                {alert.message ??
                                  alert.description ??
                                  alert.alert ??
                                  JSON.stringify(alert)}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </article>
                </section>

                {/* Historical records table */}
                <section className="panel history-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Recent Historical Records</h2>
                      <p>
                        Latest observations received from the history API
                      </p>
                    </div>
                  </div>

                  {history.length === 0 ? (
                    <p className="empty-state">
                      No history records returned.
                    </p>
                  ) : (
                    <div className="table-wrapper">
                      <table>
                        <thead>
                          <tr>
                            <th>Date</th>
                            <th>Water Availability</th>
                          </tr>
                        </thead>

                        <tbody>
                          {[...history]
                            .slice(-10)
                            .reverse()
                            .map((item, index) => {
                              const value = getPercent(item);

                              return (
                                <tr
                                  key={`${item.date ?? "record"}-${index}`}
                                >
                                  <td>
                                    {String(item.date ?? "—").slice(
                                      0,
                                      10
                                    )}
                                  </td>

                                  <td>
                                    {value === null
                                      ? "—"
                                      : `${value.toFixed(2)}%`}
                                  </td>
                                </tr>
                              );
                            })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>

                <footer>
                  AquaPredict · ML-based water availability assessment
                  <br />
                  Predictions support analysis and should not replace
                  official water-resource assessments.
                </footer>
              </>
            ) : activeTab === "performance" ? (
              <ModelPerformance />
            ) : activeTab === "simulator" ? (
              <ScenarioSimulator currentPercent={currentPercent} />
            ) : (
              /* Insights changes according to the selected tab */
              <Insights
                history={history}
                forecast={forecast}
                activeTab={activeTab}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
