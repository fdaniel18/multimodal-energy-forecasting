import { useEffect, useState } from 'react'
import api from '../api/client'
import InferenceResults from '../components/InferenceResults'
import { MODEL_COLORS } from '../constants'

export default function Inference() {
  const [models, setModels] = useState([])
  const [selected, setSelected] = useState([])
  const [buildings, setBuildings] = useState([])
  const [building, setBuilding] = useState('')
  const [days, setDays] = useState([])
  const [day, setDay] = useState('')

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function load() {
      try {
        const [m, b] = await Promise.all([api.get('/models'), api.get('/buildings')])
        // inferenta live doar pentru modelele globale + SOTA (per-cladire e doar in comparatie)
        const live = m.data.filter((x) => x.live)
        setModels(live)
        setSelected(live.filter((x) => x.checkpoint_available).map((x) => x.name))
        setBuildings(b.data)
      } catch {
        setError('Backend-ul nu răspunde. Pornește-l pe :8000.')
      }
    }
    load()
  }, [])

  useEffect(() => {
    if (!building) { setDays([]); setDay(''); return }
    setResult(null)
    api.get(`/buildings/${building}/test-days`)
      .then((r) => { setDays(r.data); setDay(r.data[0] || '') })
      .catch(() => { setDays([]); setDay('') })
  }, [building])

  function toggleModel(name, available) {
    if (!available) return
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]))
  }

  async function runInference() {
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await api.post('/infer', { building, day, models: selected })
      setResult(r.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Eroare la inferență.')
    } finally {
      setLoading(false)
    }
  }

  const canRun = building && day && selected.length > 0 && !loading

  return (
    <div>
      <div className="page-head">
        <h1>Inferență</h1>
      </div>

      {error && <div className="banner-warn">{error}</div>}

      <div className="panel">
        <div className="field" style={{ marginBottom: 18 }}>
          <label>Modele</label>
          <div className="model-options">
            {models.map((m) => {
              const on = selected.includes(m.name)
              const disabled = !m.checkpoint_available
              return (
                <span key={m.name}
                      className={`model-chip ${on ? 'on' : ''} ${disabled ? 'disabled' : ''}`}
                      onClick={() => toggleModel(m.name, m.checkpoint_available)}>
                  <span className="swatch" style={{ background: MODEL_COLORS[m.name] || '#888' }} />
                  {m.label}
                  {disabled && <span className="tag">(neantrenat)</span>}
                </span>
              )
            })}
          </div>
        </div>

        <div className="form-grid">
          <div className="field">
            <label>Clădire</label>
            <select value={building} onChange={(e) => setBuilding(e.target.value)}>
              <option value="">— alege clădirea —</option>
              {buildings.map((b) => (
                <option key={b.building_id} value={b.building_id}>
                  {b.building_id}{b.usage ? ` · ${b.usage}` : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Zi</label>
            <select value={day} onChange={(e) => setDay(e.target.value)}
                    disabled={!building || days.length === 0}>
              {!building && <option>— alege întâi clădirea —</option>}
              {building && days.length === 0 && <option>— fără zile (rulează celula 12c) —</option>}
              {days.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
        </div>

        <div className="run-row">
          <button className="btn" disabled={!canRun} onClick={runInference}>
            {loading ? 'Se rulează…' : 'Rulează inferența'}
          </button>
          {selected.length === 0 && <span className="hint">Alege cel puțin un model.</span>}
        </div>
      </div>

      {loading && <div className="progress" />}

      {result && <InferenceResults result={result} />}
    </div>
  )
}
