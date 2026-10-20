import { useEffect, useRef, useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from 'recharts'
import api from '../api/client'
import { MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, num3 } from '../constants'

const AXIS = { fontSize: 11, fill: '#7D7D7D' }
const BLUE = '#2f6df0'

const dayTick = (t) => String(t).slice(5, 10)
const fullTs = (t) => String(t).slice(0, 16)


function rmseGain(metrics, notextKey, textKey) {
  const a = metrics[notextKey]?.rmse
  const b = metrics[textKey]?.rmse
  if (a == null || b == null || a === 0) return null
  return ((a - b) / a) * 100
}


const DYN_EXAMPLES = [
  ['Zi de lucru, mijloc de săptămână',
   'This is a mid-week workday: full occupancy and normal operational loads expected. Recent outdoor temperature (last 24 hours): cold, around 2 degrees. Next 24h weather forecast: cold and partly cloudy, with a slight warming trend. Consumption over the last week has been stable, with a clear daily peak in the afternoon.'],
  ['Weekend',
   'This is a weekend day: reduced occupancy and lower operational loads expected. Recent outdoor temperature (last 24 hours): mild, around 14 degrees. Next 24h weather forecast: mild and mostly clear. Consumption over the last week is clearly lower at the weekend than on weekdays.'],
  ['Sărbătoare legală',
   'This is a public holiday: the building is mostly closed, with only a minimal baseline load expected. Next 24h weather forecast: cold and overcast. Consumption is expected to stay near the overnight baseline for most of the day.'],
]

export default function Building() {
  const [buildings, setBuildings] = useState([])
  const [id, setId] = useState('')
  const [info, setInfo] = useState(null)
  const [history, setHistory] = useState([])
  const [profile, setProfile] = useState(null)
  const [preds, setPreds] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    api.get('/buildings').then((r) => setBuildings(r.data))
      .catch(() => setError('Backend-ul nu răspunde.'))
  }, [])

  useEffect(() => {
    if (!id) { setInfo(null); setHistory([]); setProfile(null); setPreds([]); return }
    Promise.all([
      api.get(`/buildings/${id}`),
      api.get(`/buildings/${id}/history?resample=D`),
      api.get(`/buildings/${id}/profile`),
      api.get(`/buildings/${id}/predictions`),
    ]).then(([i, h, p, pr]) => {
      setInfo(i.data); setHistory(h.data); setProfile(p.data); setPreds(pr.data)
    }).catch(() => setError('Nu s-au putut încărca datele clădirii.'))
  }, [id])

  const metrics = info?.metrics || {}
  const metricModels = MODEL_ORDER.filter((m) => metrics[m])
  const predModels = preds.length
    ? MODEL_ORDER.filter((m) => preds[0][m] !== undefined)
    : []
  const gainPerhouse = rmseGain(metrics, 'perhouse_notext', 'perhouse_text')
  const gainGlobal = rmseGain(metrics, 'global_notext', 'global_text')

  return (
    <div>
      <div className="page-head">
        <h1>Clădire</h1>
        <p>Istoricul de consum, profilele și metricile pentru o clădire.</p>
      </div>

      {error && <div className="banner-warn">{error}</div>}

      <div className="panel">
        <div className="field">
          <label>Clădire</label>
          <select value={id} onChange={(e) => setId(e.target.value)}>
            <option value="">— alege clădirea —</option>
            {buildings.map((b) => (
              <option key={b.building_id} value={b.building_id}>
                {b.building_id}{b.usage ? ` · ${b.usage}` : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {info && (
        <>
          <div className="info-grid">
            <div className="info-card"><div className="k">Tip</div><div className="v">{info.usage || '—'}</div></div>
            <div className="info-card"><div className="k">Suprafață</div><div className="v">{info.sqm ? `${Math.round(info.sqm)} m²` : '—'}</div></div>
            <div className="info-card"><div className="k">An construcție</div><div className="v">{info.yearbuilt || '—'}</div></div>
            <div className="info-card"><div className="k">Site</div><div className="v">{info.site || '—'}</div></div>
            <div className="info-card"><div className="k">Consum mediu</div><div className="v">{info.mean_consumption != null ? `${info.mean_consumption.toFixed(1)} kWh` : '—'}</div></div>
          </div>

          <div className="result-section">
            <h2>Istoric consum (medie zilnică)</h2>
            <div className="chart-box">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={history} margin={{ top: 6, right: 16, bottom: 6, left: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="ts" tick={AXIS} minTickGap={60}
                         tickFormatter={(t) => String(t).slice(0, 7)} />
                  <YAxis tick={AXIS} tickFormatter={num3} />
                  <Tooltip labelFormatter={(t) => String(t).slice(0, 10)} formatter={num3} />
                  <Line type="monotone" dataKey="value" name="kWh" stroke={BLUE}
                        strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {preds.length > 1 && predModels.length > 0 && (
            <div className="result-section">
              <h2>Predicții pe setul de test</h2>
              <PredictionsChart preds={preds} predModels={predModels} />
            </div>
          )}

          {profile && (
            <div className="result-section">
              <h2>Profile de consum</h2>
              <div className="chart-grid">
                <div className="chart-box">
                  <h3>Pe zilele săptămânii</h3>
                  <ResponsiveContainer width="100%" height={230}>
                    <BarChart data={profile.weekday}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis dataKey="weekday" tick={AXIS} interval={0} angle={-25} textAnchor="end" height={50} />
                      <YAxis tick={AXIS} tickFormatter={num3} />
                      <Tooltip formatter={num3} />
                      <Bar dataKey="mean" name="kWh" fill={BLUE} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-box">
                  <h3>Pe luni</h3>
                  <ResponsiveContainer width="100%" height={230}>
                    <BarChart data={profile.month}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis dataKey="month" tick={AXIS} interval={0} />
                      <YAxis tick={AXIS} tickFormatter={num3} />
                      <Tooltip formatter={num3} />
                      <Bar dataKey="mean" name="kWh" fill={MODEL_COLORS.patchtst} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="chart-box">
                  <h3>Pe ore</h3>
                  <ResponsiveContainer width="100%" height={230}>
                    <LineChart data={profile.hour}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis dataKey="hour" tick={AXIS} />
                      <YAxis tick={AXIS} tickFormatter={num3} />
                      <Tooltip formatter={num3} />
                      <Line type="monotone" dataKey="mean" name="kWh" stroke="#f2a13a" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {metricModels.length > 0 && (
            <div className="result-section">
              <h2>Metrici (pe tot setul de test)</h2>
              <table className="data-table">
                <thead>
                  <tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>WAPE</th></tr>
                </thead>
                <tbody>
                  {metricModels.map((m) => (
                    <tr key={m}>
                      <td><span className="swatch" style={{ background: MODEL_COLORS[m] }} />{MODEL_LABELS[m]}</td>
                      <td>{metrics[m].rmse.toFixed(2)}</td>
                      <td>{metrics[m].mae.toFixed(2)}</td>
                      <td>{metrics[m].r2.toFixed(3)}</td>
                      <td>{(metrics[m].wape * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {(gainPerhouse != null || gainGlobal != null) && (
                <div style={{ marginTop: 12, lineHeight: 1.8 }}>
                  {gainPerhouse != null && <GainLine label="Per clădire" gain={gainPerhouse} />}
                  {gainGlobal != null && <GainLine label="Global" gain={gainGlobal} />}
                </div>
              )}
            </div>
          )}

          <div className="result-section">
            <h2>Descrierea text a clădirii</h2>
            {info.text_static ? (
              <>
                <h3>Context static (identitatea clădirii)</h3>
                <div className="text-box">{info.text_static}</div>
              </>
            ) : (
              <div className="note">Textul static nu a fost încă exportat pentru această clădire.</div>
            )}
            <h3 style={{ marginTop: 18 }}>Exemple de context dinamic</h3>
            {DYN_EXAMPLES.map(([when, text]) => (
              <div key={when} style={{ marginTop: 12 }}>
                <div style={{ fontWeight: 600, marginBottom: 4, color: '#252525' }}>{when}</div>
                <div className="text-box">{text}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function PredictionsChart({ preds, predModels }) {
  const [range, setRange] = useState([0, preds.length - 1])
  const boxRef = useRef(null)

  const [start, end] = range
  const data = preds.slice(start, end + 1)
  const zoomed = start > 0 || end < preds.length - 1

  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    function onWheel(e) {
      e.preventDefault()
      setRange(([s, ee]) => {
        const span = ee - s
        const factor = e.deltaY < 0 ? 0.8 : 1.25
        let newSpan = Math.round(span * factor)
        newSpan = Math.max(23, Math.min(preds.length - 1, newSpan))
        const center = (s + ee) / 2
        let ns = Math.round(center - newSpan / 2)
        let nee = ns + newSpan
        if (ns < 0) { ns = 0; nee = newSpan }
        if (nee > preds.length - 1) { nee = preds.length - 1; ns = nee - newSpan }
        return [Math.max(0, ns), nee]
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [preds.length])

  function reset() { setRange([0, preds.length - 1]) }

  return (
    <>
      <div className="toolbar" style={{ justifyContent: 'space-between' }}>
        <span style={{ color: 'var(--mid)', fontSize: '0.85rem' }}>
          Rotește rotița mouse-ului peste grafic ca să faci zoom.
        </span>
        {zoomed && <button className="btn" onClick={reset}>Resetează zoom</button>}
      </div>
      <div className="chart-box" ref={boxRef}>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={data} margin={{ top: 6, right: 16, bottom: 6, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="ts" tick={AXIS} minTickGap={48} tickFormatter={dayTick} />
            <YAxis tick={AXIS} tickFormatter={num3}
                   label={{ value: 'kWh', angle: -90, position: 'insideLeft', fill: '#7D7D7D', fontSize: 11 }} />
            <Tooltip labelFormatter={fullTs} formatter={num3} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="real" name="Real" stroke={MODEL_COLORS.real}
                  strokeWidth={2.2} dot={false} isAnimationActive={false} />
            {predModels.map((m) => (
              <Line key={m} type="monotone" dataKey={m} name={MODEL_LABELS[m] || m}
                    stroke={MODEL_COLORS[m] || '#888'} strokeWidth={1.4} dot={false}
                    isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  )
}

function GainLine({ label, gain }) {
  const better = gain > 0
  return (
    <div style={{ color: '#252525' }}>
      <strong>{label}</strong> — textul {better ? 'reduce' : 'crește'} RMSE cu{' '}
      <strong>{Math.abs(gain).toFixed(1)}%</strong>
    </div>
  )
}
