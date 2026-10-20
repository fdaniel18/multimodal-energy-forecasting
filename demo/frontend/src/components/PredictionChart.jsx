import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { MODEL_COLORS, MODEL_LABELS, num3 } from '../constants'

export default function PredictionChart({ result }) {
  const modelNames = Object.keys(result.models)
  const data = result.hours.map((h, i) => {
    const row = { ora: h, [MODEL_LABELS.real]: result.real[i] }
    modelNames.forEach((m) => {
      row[MODEL_LABELS[m] || m] = result.models[m].pred[i]
    })
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={360}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 16, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="ora" tick={{ fontSize: 12, fill: '#7D7D7D' }}
               label={{ value: 'Ora', position: 'insideBottom', offset: -6, fill: '#7D7D7D' }} />
        <YAxis tick={{ fontSize: 12, fill: '#7D7D7D' }} tickFormatter={num3}
               label={{ value: 'kWh', angle: -90, position: 'insideLeft', fill: '#7D7D7D' }} />
        <Tooltip formatter={num3} />
        <Legend />
        <Line type="monotone" dataKey={MODEL_LABELS.real} stroke={MODEL_COLORS.real}
              strokeWidth={2.5} dot={false} />
        {modelNames.map((m) => (
          <Line key={m} type="monotone" dataKey={MODEL_LABELS[m] || m}
                stroke={MODEL_COLORS[m] || '#888'} strokeWidth={2} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
