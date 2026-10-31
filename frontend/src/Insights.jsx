
import "./Insights.css";

function getValue(item) {
  const value =
    item.percent_full ??
    item.predicted_percent_full ??
    item.prediction;

  return value !== undefined &&
    value !== null &&
    Number.isFinite(Number(value))
    ? Number(value)
    : null;
}

function average(values) {
  if (values.length === 0) return null;

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getMonthName(month) {
  return new Date(2024, month, 1).toLocaleString("en", {
    month: "short",
  });
}

export default function Insights({ history, forecast, activeTab }) {
  const observed = history
    .map((item) => ({
      ...item,
      value: getValue(item),
    }))
    .filter((item) => item.value !== null)
    .sort((a, b) =>
      String(a.date).localeCompare(String(b.date))
    );

  const recent = observed.slice(-30);
  const previous = observed.slice(-60, -30);

  const recentAverage = average(
    recent.map((item) => item.value)
  );

  const previousAverage = average(
    previous.map((item) => item.value)
  );

  // Indicative score based on the latest 30 observations.
  const sustainabilityScore =
    recentAverage === null
      ? null
      : Math.round(
          Math.max(0, Math.min(100, recentAverage))
        );

  const trend =
    recentAverage === null || previousAverage === null
      ? null
      : recentAverage - previousAverage;

  // Calculate average water availability for each calendar month.
  const monthlyData = Array.from({ length: 12 }, (_, month) => {
    const values = observed
      .filter((item) => {
        if (!item.date) return false;

        return (
          new Date(`${item.date}T00:00:00`).getMonth() === month
        );
      })
      .map((item) => item.value);

    return {
      month: getMonthName(month),
      value: average(values),
      count: values.length,
    };
  });

  const forecastValues = forecast
    .map((item) => ({
      ...item,
      value: getValue(item),
    }))
    .filter((item) => item.value !== null);

  const latest = observed.length
    ? observed[observed.length - 1]
    : null;

  // Illustrative threshold, not an official reservoir standard.
  const threshold = 30;

  const riskValue =
    latest && forecastValues.length
      ? Math.min(
          latest.value,
          ...forecastValues.map((item) => item.value)
        )
      : latest?.value ?? null;

  let risk = "Insufficient data";
  let riskClass = "risk-unknown";

  if (riskValue !== null) {
    if (riskValue < threshold) {
      risk = "High";
      riskClass = "risk-high";
    } else if (riskValue < 50) {
      risk = "Moderate";
      riskClass = "risk-moderate";
    } else {
      risk = "Lower";
      riskClass = "risk-low";
    }
  }

  return (
    <div className="insights-page">

      {/* 1. WATER SUSTAINABILITY SCORE */}
      {activeTab === "sustainability" && (
        <section className="insight-card">
          <p className="insight-eyebrow">
            WATER SUSTAINABILITY
          </p>

          <h2>Water Sustainability Score</h2>

          <div className="score-layout">
            <div className="score-circle">
              <strong>
                {sustainabilityScore === null
                  ? "—"
                  : sustainabilityScore}
              </strong>

              <span>out of 100</span>
            </div>

            <div>
              <h3>
                {sustainabilityScore === null
                  ? "Data unavailable"
                  : sustainabilityScore >= 70
                    ? "Higher current availability"
                    : sustainabilityScore >= 40
                      ? "Moderate current availability"
                      : "Lower current availability"}
              </h3>

              <p>
                This indicative score is based on the average
                recorded reservoir fullness over the latest
                30 observations. It is not an official
                sustainability assessment.
              </p>

              <p>
                <strong>30-observation average:</strong>{" "}
                {recentAverage === null
                  ? "—"
                  : `${recentAverage.toFixed(2)}%`}
              </p>
            </div>
          </div>
        </section>
      )}

      {/* 2. SEASONAL AND LONG-TERM TREND ANALYSIS */}
      {activeTab === "trends" && (
        <section className="insight-card">
          <p className="insight-eyebrow">
            HISTORICAL PATTERNS
          </p>

          <h2>Seasonal and Long-Term Trend Analysis</h2>

          <p>
            Monthly averages show how recorded water availability
            varies across the calendar year. They combine
            observations from different years, so they do not
            represent a forecast.
          </p>

          <div className="monthly-chart">
            {monthlyData.map((item) => (
              <div className="month-column" key={item.month}>
                <strong>
                  {item.value === null
                    ? "—"
                    : `${item.value.toFixed(0)}%`}
                </strong>

                <div className="month-track">
                  <div
                    className="month-fill"
                    style={{
                      height: `${Math.max(
                        0,
                        Math.min(100, item.value ?? 0)
                      )}%`,
                    }}
                  />
                </div>

                <span>{item.month}</span>
              </div>
            ))}
          </div>

          <div className="trend-summary">
            <div>
              <span>Latest 30 observations</span>

              <strong>
                {recentAverage === null
                  ? "—"
                  : `${recentAverage.toFixed(2)}%`}
              </strong>
            </div>

            <div>
              <span>Previous 30 observations</span>

              <strong>
                {previousAverage === null
                  ? "—"
                  : `${previousAverage.toFixed(2)}%`}
              </strong>
            </div>

            <div>
              <span>Recent change</span>

              <strong>
                {trend === null
                  ? "Not enough data"
                  : `${trend > 0 ? "+" : ""}${trend.toFixed(2)} percentage points`}
              </strong>
            </div>
          </div>

          <p className="insight-note">
            The recent change compares two consecutive
            30-observation periods. It is a short-term indicator,
            not proof of a long-term climate trend.
          </p>
        </section>
      )}

      {/* 3. WATER SCARCITY RISK ASSESSMENT */}
      {activeTab === "scarcity" && (
        <section className="insight-card">
          <p className="insight-eyebrow">
            EARLY WARNING
          </p>

          <h2>Water Scarcity Risk Assessment</h2>

          <div className={`risk-banner ${riskClass}`}>
            <div>
              <span>Indicative availability risk</span>
              <h3>{risk}</h3>
            </div>

            <div className="risk-value">
              {riskValue === null
                ? "—"
                : `${riskValue.toFixed(2)}%`}
            </div>
          </div>

          <p>
            This assessment uses the latest observed fullness
            and the lowest value in the available forecast.
            The 30% threshold is illustrative and should be
            replaced with an appropriate reservoir-specific
            threshold before operational use.
          </p>

          <div className="risk-details">
            <div>
              <span>Latest observed</span>

              <strong>
                {latest
                  ? `${latest.value.toFixed(2)}%`
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Lowest forecast</span>

              <strong>
                {forecastValues.length
                  ? `${Math.min(
                      ...forecastValues.map(
                        (item) => item.value
                      )
                    ).toFixed(2)}%`
                  : "—"}
              </strong>
            </div>

            <div>
              <span>Illustrative threshold</span>
              <strong>{threshold}%</strong>
            </div>
          </div>

          <p className="insight-note">
            This is a screening indicator based on water
            availability, not a probability of scarcity or a
            substitute for official water-resource decisions.
          </p>
        </section>
      )}
    </div>
  );
}
