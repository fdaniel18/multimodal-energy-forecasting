import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import api from '../api/client'
import { MODEL_COLORS, MODEL_LABELS, num3 } from '../constants'

export default function Home() {
  const [board, setBoard] = useState(null)
  const [teaser, setTeaser] = useState(null)
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    async function load() {
      try {
        const overview = await api.get('/models/stats')
        const rows = overview.data
          .map((m) => ({ model: m.model, label: m.label, rmse: m.stats?.rmse?.median }))
          .filter((r) => r.rmse != null)
          .sort((a, b) => a.rmse - b.rmse)
        setBoard(rows)
      } catch {
        setOffline(true)
        return
      }
      try {
        const blist = await api.get('/buildings')
        const bid = blist.data?.[0]?.building_id
        if (bid) {
          const preds = await api.get(`/buildings/${bid}/predictions`, { params: { limit: 96 } })
          setTeaser({ bid, rows: preds.data })
        }
      } catch { }
    }
    load()
  }, [])

  const maxRmse = board && board.length ? Math.max(...board.map((r) => r.rmse)) : 1

  return (
    <div>
      {offline && (
        <div className="banner-warn">
          Backend-ul nu răspunde. Pornește-l cu{' '}
          <code>uvicorn demo.backend.app:app --port 8000</code> și reîncarcă pagina.
        </div>
      )}

      <div className="intro">
        <h1>Predicția consumului energetic cu context textual</h1>
      </div>

      {board && board.length > 0 && (
        <section className="home-section" style={{ maxWidth: 760, margin: '0 auto' }}>
          <div className="home-head" style={{ textAlign: 'center' }}>
            <h2>RMSE median pe orizontul de 24h</h2>
          </div>
          <div className="leaderboard">
            {board.map((r, i) => (
              <div className="lb-row" key={r.model}>
                <span className="lb-rank">{i + 1}</span>
                <span className="lb-label">
                  <span className="swatch" style={{ background: MODEL_COLORS[r.model] || '#888' }} />
                  {r.label || MODEL_LABELS[r.model] || r.model}
                </span>
                <span className="lb-bar">
                  <span
                    className="lb-fill"
                    style={{
                      width: `${Math.max(6, (r.rmse / maxRmse) * 100)}%`,
                      background: MODEL_COLORS[r.model] || '#888',
                    }}
                  />
                </span>
                <span className="lb-val">{r.rmse.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {teaser && teaser.rows && teaser.rows.length > 1 && (
        <section className="home-section">
          <div className="home-head">
            <h2>Clădirea {teaser.bid} — consum real vs. modele</h2>
          </div>
          <div className="chart-box" style={{ padding: '14px 12px 6px' }}>
            <TeaserChart rows={teaser.rows} />
          </div>
        </section>
      )}
    </div>
  )
}

function fmtHour(ts) {
  if (!ts) return ''
  const d = new Date(String(ts).replace(' ', 'T'))
  const p = (n) => String(n).padStart(2, '0')
  return `${p(d.getDate())}.${p(d.getMonth() + 1)} ${p(d.getHours())}:00`
}

function TeaserChart({ rows }) {
  const present = Object.keys(rows[0]).filter((k) => k !== 'ts' && k !== 'real')
  const preferred = ['global_text', 'patchtst', 'global_notext']
  const picked = preferred.filter((m) => present.includes(m)).slice(0, 2)
  const lines = picked.length ? picked : present.slice(0, 2)

  const data = rows.map((row) => {
    const o = { ts: row.ts, Real: row.real }
    lines.forEach((m) => { o[MODEL_LABELS[m] || m] = row[m] })
    return o
  })

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 6, right: 16, bottom: 4, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
        <XAxis dataKey="ts" tickFormatter={fmtHour} tick={{ fontSize: 11, fill: '#7D7D7D' }}
               tickLine={false} minTickGap={56} />
        <YAxis tick={{ fontSize: 11, fill: '#7D7D7D' }} tickLine={false} width={44} tickFormatter={num3}
               label={{ value: 'kWh', angle: -90, position: 'insideLeft', fill: '#7D7D7D', fontSize: 11 }} />
        <Tooltip formatter={num3} labelFormatter={fmtHour} />
        <Line type="monotone" dataKey="Real" stroke={MODEL_COLORS.real} strokeWidth={2.4} dot={false} />
        {lines.map((m) => (
          <Line key={m} type="monotone" dataKey={MODEL_LABELS[m] || m}
                stroke={MODEL_COLORS[m] || '#888'} strokeWidth={1.8} dot={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
