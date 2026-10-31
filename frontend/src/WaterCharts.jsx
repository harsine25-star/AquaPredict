import { useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

function getPercentage(item) {
  const value =
    item.percent_full ??
    item.predicted_percent_full ??
    item.prediction ??
    item.predicted ??
    item.percentage;

  if (value === undefined || value === null || value === "") {
    return null;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function prepareData(items, isForecast = false) {
  return items
    .map((item, index) => ({
      date:
        item.date ??
        item.forecast_date ??
        item.prediction_date ??
        `Day ${index + 1}`,
      percentage: getPercentage(item),
    }))
    .filter((item) => item.percentage !== null)
    .map((item) => ({
      ...item,
      date: String(item.date).slice(0, 10),
    }));
}

function ChartCard({ title, subtitle, data, color, showThreshold = false }) {
  return (
    <section className="panel water-chart-panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <span className="panel-tag">
          {data.length} records
        </span>
      </div>

      {data.length === 0 ? (
        <p className="empty-state">
          No valid percentage data is available for this chart.
        </p>
      ) : (
        <div className="water-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 10, right: 15, left: 0, bottom: 10 }}
            >
              <CartesianGrid
                strokeDasharray="4 4"
                stroke="#e8eef5"
              />

              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "#718096" }}
                minTickGap={25}
              />

              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 11, fill: "#718096" }}
                tickFormatter={(value) => `${value}%`}
                width={48}
              />

              <Tooltip
                formatter={(value) => [
                  `${Number(value).toFixed(2)}%`,
                  "Reservoir full",
                ]}
                labelFormatter={(label) => `Date: ${label}`}
                contentStyle={{
                  borderRadius: 10,
                  border: "1px solid #dce6f0",
                  fontSize: 12,
                }}
              />

              <Legend />

              {showThreshold && (
                <ReferenceLine
                  y={30}
                  stroke="#e69a32"
                  strokeDasharray="5 5"
                  label={{
                    value: "Illustrative 30% threshold",
                    position: "insideTopRight",
                    fontSize: 10,
                    fill: "#a66b15",
                  }}
                />
              )}

              <Line
                type="monotone"
                dataKey="percentage"
                name="Water availability"
                stroke={color}
                strokeWidth={2.5}
                dot={data.length <= 60}
                activeDot={{ r: 6 }}
                connectNulls={false}
                isAnimationActive
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <p className="chart-note">
        Hover over the graph to inspect individual values.
      </p>
    </section>
  );
}

export default function WaterCharts({ history, forecast }) {
  const [range, setRange] = useState("all");

  const historicalData = prepareData(history);
  const forecastData = prepareData(forecast, true);

  const cutoff = new Date();
  cutoff.setHours(0, 0, 0, 0);

  const filteredHistory = historicalData.filter((item) => {
    const date = new Date(item.date + "T00:00:00");

    if (Number.isNaN(date.getTime())) return false;

    if (range === "30") {
      const start = new Date(cutoff);
      start.setDate(start.getDate() - 30);
      return date >= start && date <= cutoff;
    }

    if (range === "365") {
      const start = new Date(cutoff);
      start.setDate(start.getDate() - 365);
      return date >= start && date <= cutoff;
    }

    return date <= cutoff;
  });

  return (
    <div className="charts-section">
      <div className="chart-filter">
        <div>
          <h2>Water Trends & Predictions</h2>
          <p>Explore recorded history and upcoming estimates.</p>
        </div>

        <select
          aria-label="Historical time range"
          value={range ?? "all"}
          onChange={(event) =>
            setRange(event.target.value)
          }
        >
          <option value="30">Last 30 days</option>
          <option value="365">Last year</option>
          <option value="all">All history</option>
        </select>
      </div>

      <ChartCard
        title="Historical Water Availability"
        subtitle="Recorded reservoir percentage full"
        data={filteredHistory}
        color="#1687c9"
      />

      <ChartCard
        title="Future Water Availability"
        subtitle="Machine-learning forecast for the next seven days"
        data={forecastData}
        color="#13a88a"
        showThreshold
      />
    </div>
  );
}
