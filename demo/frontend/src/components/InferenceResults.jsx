import PredictionChart from './PredictionChart'
import { MODEL_COLORS, MODEL_LABELS } from '../constants'

const METRIC_COLS = [
  ['rmse', 'RMSE'], ['mae', 'MAE'], ['r2', 'R²'], ['wape', 'WAPE'],
]

export default function InferenceResults({ result }) {
  const modelNames = Object.keys(result.models || {})
  if (modelNames.length === 0) return <div className="note">Niciun model în rezultat.</div>
  const bestRmse = Math.min(...modelNames.map((m) => result.models[m].metrics.rmse))

  return (
    <div>
      <div className="result-section">
        <h2>{result.building} · {result.day}</h2>
        <div className="chart-wrap"><PredictionChart result={result} /></div>
        {result.skipped?.length > 0 && (
          <div className="note">Sărit (checkpoint indisponibil): {result.skipped.join(', ')}</div>
        )}
      </div>

      <div className="result-section">
        <h2>Metrici (pe ziua aleasă)</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              {METRIC_COLS.map(([, lab]) => <th key={lab}>{lab}</th>)}
              <th>Timp (ms)</th>
            </tr>
          </thead>
          <tbody>
            {modelNames.map((m) => (
              <tr key={m}>
                <td>
                  <span className="swatch" style={{ background: MODEL_COLORS[m] }} />
                  {MODEL_LABELS[m] || m}
                </td>
                {METRIC_COLS.map(([key]) => {
                  const v = result.models[m].metrics[key]
                  const best = key === 'rmse' && v === bestRmse
                  return (
                    <td key={key} style={best ? { fontWeight: 700 } : undefined}>
                      {v == null ? '—' : key === 'wape' ? `${(v * 100).toFixed(1)}%` : v.toFixed(2)}
                    </td>
                  )
                })}
                <td>{result.models[m].infer_ms != null ? result.models[m].infer_ms.toFixed(1) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {result.infer_ms_total != null && (
          <div className="note">
            Timp total de inferență pe această zi {result.infer_ms_total.toFixed(1)} ms, pe CPU,
            fără timpul de încărcare a modelelor (acela se face o singură dată, la pornirea backendului).
          </div>
        )}
      </div>

      <div className="result-section">
        <h2>Valori orare (kWh)</h2>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Ora</th>
                <th>Real</th>
                {modelNames.map((m) => <th key={m}>{MODEL_LABELS[m] || m}</th>)}
              </tr>
            </thead>
            <tbody>
              {result.hours.map((h, i) => (
                <tr key={h}>
                  <td>{h}:00</td>
                  <td>{result.real[i].toFixed(1)}</td>
                  {modelNames.map((m) => <td key={m}>{result.models[m].pred[i].toFixed(1)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {result.models.hybrid && (
          <div className="note">
            Contribuția medie a textului (model cu corecție):{' '}
            {(result.models.hybrid.text_offset.reduce((a, b) => a + b, 0)
              / result.models.hybrid.text_offset.length).toFixed(2)} kWh/oră.
          </div>
        )}
      </div>

      {result.text_static && (
        <div className="result-section">
          <h2>Descrierea folosită de model</h2>
          <div className="text-box">{result.text_static}</div>
        </div>
      )}
    </div>
  )
}
