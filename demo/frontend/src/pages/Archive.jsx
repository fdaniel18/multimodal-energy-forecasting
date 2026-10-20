import { useEffect, useState } from 'react'
import api from '../api/client'
import InferenceResults from '../components/InferenceResults'
import { MODEL_LABELS } from '../constants'

function fmtDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('ro-RO', { dateStyle: 'short', timeStyle: 'short' })
}

export default function Archive() {
  const [runs, setRuns] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.get('/inference/runs')
      .then((r) => setRuns(r.data))
      .catch(() => setError('Nu s-a putut încărca arhiva.'))
  }, [])

  function openRun(id) {
    setSelectedId(id)
    setDetail(null)
    api.get(`/inference/runs/${id}`)
      .then((r) => setDetail(r.data.result))
      .catch(() => setError('Nu s-a putut încărca rularea selectată.'))
  }

  return (
    <div>
      <div className="page-head">
        <h1>Arhivă</h1>
        <p>Inferențele rulate anterior. Apasă o rulare pentru a o revedea.</p>
      </div>

      {error && <div className="banner-warn">{error}</div>}

      {runs.length === 0 && !error && (
        <div className="placeholder">
          Nicio rulare încă. Rulează o inferență în pagina Inferență și va apărea aici.
        </div>
      )}

      {runs.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Data</th>
              <th>Clădire</th>
              <th>Zi prezisă</th>
              <th>Modele</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} className={`clickable ${r.id === selectedId ? 'row-selected' : ''}`}
                  onClick={() => openRun(r.id)}>
                <td>{fmtDate(r.created_at)}</td>
                <td>{r.building_id}</td>
                <td>{r.day}</td>
                <td>{r.model_names.map((m) => MODEL_LABELS[m] || m).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {detail && (
        <div style={{ marginTop: 28 }}>
          <InferenceResults result={detail} />
        </div>
      )}
    </div>
  )
}
