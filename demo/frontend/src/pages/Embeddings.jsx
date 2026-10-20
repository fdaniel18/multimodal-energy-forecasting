import { useEffect, useState, useMemo } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
import api from '../api/client'
import { CATEGORICAL } from '../constants'

const Plot = createPlotlyComponent(Plotly)

function wrap(s, width = 46) {
  if (!s) return ''
  const words = String(s).split(/\s+/)
  const lines = []
  let line = ''
  for (const w of words) {
    if ((line + ' ' + w).trim().length > width) { lines.push(line.trim()); line = w }
    else line += ' ' + w
  }
  if (line.trim()) lines.push(line.trim())
  return lines.join('<br>')
}

function hoverText(p) {
  const head = `<b>${p.building_id}</b>`
  const type = p.sub_usage && p.sub_usage !== p.usage
    ? `${p.usage} · ${p.sub_usage}` : (p.usage || 'necunoscut')
  const gain = p.gain != null
    ? `<br>textul schimbă RMSE cu ${p.gain > 0 ? '−' : '+'}${Math.abs(p.gain).toFixed(1)}%` : ''
  const meta = [type, p.site ? `site ${p.site}` : null, `cluster ${p.cluster}`]
    .filter(Boolean).join(' · ')
  const desc = p.text_static
    ? `<br><span style="color:#CFCFCF">${wrap(p.text_static)}</span>` : ''
  return `${head}<br>${meta}${gain}${desc}`
}

function buildGains(perBuilding) {
  const byB = {}
  perBuilding.forEach((r) => {
    byB[r.building_id] = byB[r.building_id] || {}
    byB[r.building_id][r.model] = r.rmse
  })
  const gains = {}
  Object.entries(byB).forEach(([b, m]) => {
    let nt = m.global_notext, t = m.global_text
    if (nt == null || t == null) { nt = m.perhouse_notext; t = m.perhouse_text }
    if (nt != null && t != null && nt > 0) {
      gains[b] = { gain: ((nt - t) / nt) * 100, ntRmse: nt, tRmse: t }
    }
  })
  return gains
}

const pct = (v) => `${v > 0 ? '−' : '+'}${Math.abs(v).toFixed(1)}%`

export default function Embeddings() {
  const [points, setPoints] = useState(null)
  const [colorBy, setColorBy] = useState('usage')
  const [notReady, setNotReady] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([api.get('/embeddings'), api.get('/metrics')])
      .then(([e, m]) => {
        const gains = buildGains(m.data?.per_building || [])
        setPoints(e.data.map((p) => {
          const g = gains[p.building_id]
          return { ...p, gain: g ? g.gain : null, ntRmse: g?.ntRmse ?? null, tRmse: g?.tRmse ?? null }
        }))
      })
      .catch((err) => {
        if (err.response?.status === 404) setNotReady(true)
        else setError('Backend-ul nu răspunde.')
      })
  }, [])

  const is3d = points && points.length > 0 && points[0].z != null
  const withGain = useMemo(
    () => (points || []).filter((p) => p.gain != null), [points])

  const traces = useMemo(() => {
    if (!is3d) return []
    if (colorBy === 'gain') {
      const g = points.filter((p) => p.gain != null)
      const ng = points.filter((p) => p.gain == null)
      const out = []
      if (g.length) out.push({
        type: 'scatter3d', mode: 'markers', name: 'cu metrici',
        x: g.map((p) => p.x), y: g.map((p) => p.y), z: g.map((p) => p.z),
        text: g.map((p) => hoverText(p)), hoverinfo: 'text',
        hoverlabel: { align: 'left', bgcolor: '#252525', bordercolor: '#545454',
                      font: { color: '#fff', size: 12 } },
        showlegend: false,
        marker: {
          size: 8, opacity: 0.95, line: { width: 0.5, color: '#fff' },
          color: g.map((p) => p.gain), colorscale: 'Blues', showscale: true,
          colorbar: { title: 'scădere RMSE %', thickness: 12, len: 0.55, x: 0.92 },
        },
      })
      if (ng.length) out.push({
        type: 'scatter3d', mode: 'markers', name: 'fără metrici',
        x: ng.map((p) => p.x), y: ng.map((p) => p.y), z: ng.map((p) => p.z),
        text: ng.map((p) => hoverText(p)), hoverinfo: 'text', showlegend: false,
        marker: { size: 6, color: '#cccccc', opacity: 0.6, line: { width: 0.5, color: '#fff' } },
      })
      return out
    }
    const keyOf = (p) => (colorBy === 'usage' ? (p.usage || 'necunoscut') : `Cluster ${p.cluster}`)
    const map = new Map()
    points.forEach((p) => {
      const k = keyOf(p)
      if (!map.has(k)) map.set(k, [])
      map.get(k).push(p)
    })
    return [...map.keys()].sort().map((k, i) => {
      const d = map.get(k)
      return {
        type: 'scatter3d', mode: 'markers', name: k,
        x: d.map((p) => p.x), y: d.map((p) => p.y), z: d.map((p) => p.z),
        text: d.map((p) => hoverText(p)),
        hoverinfo: 'text',
        hoverlabel: { align: 'left', bgcolor: '#252525', bordercolor: '#545454',
                      font: { color: '#fff', size: 12 } },
        marker: { size: 8, color: CATEGORICAL[i % CATEGORICAL.length], opacity: 0.9,
                  line: { width: 0.5, color: '#fff' } },
      }
    })
  }, [points, colorBy, is3d])

  const sortedGain = useMemo(
    () => [...withGain].sort((a, b) => b.gain - a.gain), [withGain])
  const top = sortedGain.slice(0, 5)
  const bottom = sortedGain.slice(-5).reverse()
  const byType = useMemo(() => {
    const m = {}
    withGain.forEach((p) => {
      const u = p.usage || 'necunoscut'
      ;(m[u] = m[u] || []).push(p.gain)
    })
    return Object.entries(m)
      .map(([usage, arr]) => ({
        usage,
        mean: arr.reduce((a, b) => a + b, 0) / arr.length,
        n: arr.length,
        help: arr.filter((g) => g > 0).length,
      }))
      .sort((a, b) => b.mean - a.mean)
  }, [withGain])
  const helpfulCount = withGain.filter((p) => p.gain > 0).length
  const meanGain = withGain.length
    ? withGain.reduce((a, p) => a + p.gain, 0) / withGain.length : 0
  const best = sortedGain[0]
  const worst = sortedGain[sortedGain.length - 1]

  return (
    <div>
      <div className="page-head">
        <h1>Embeddings</h1>
      </div>

      {error && <div className="banner-warn">{error}</div>}

      {notReady && (
        <div className="placeholder">
          Embeddings-urile nu au fost încă exportate. Rulează celula <strong>12b</strong> din notebook,
          apoi repornește backend-ul.
        </div>
      )}

      {points && !is3d && (
        <div className="banner-warn">
          Datele de embeddings sunt în format 2D vechi. Rulează din nou celula <strong>12b</strong>
          {' '}(actualizată la proiecție 3D) și repornește backend-ul.
        </div>
      )}

      {is3d && (
        <>
          <div className="toolbar">
            <span style={{ color: 'var(--mid)', fontSize: '0.88rem' }}>Colorează după:</span>
            <div className="seg">
              <button className={colorBy === 'usage' ? 'on' : ''} onClick={() => setColorBy('usage')}>Tip clădire</button>
              <button className={colorBy === 'cluster' ? 'on' : ''} onClick={() => setColorBy('cluster')}>Cluster</button>
              <button className={colorBy === 'gain' ? 'on' : ''} onClick={() => setColorBy('gain')}>Cât ajută textul</button>
            </div>
            <span style={{ color: 'var(--mid)', fontSize: '0.85rem' }}>{points.length} clădiri</span>
          </div>

          {colorBy === 'gain' && withGain.length > 0 && (
            <div className="note" style={{ marginTop: 0, marginBottom: 12 }}>
              Fiecare punct e o clădire. Cu cât albastrul e mai închis, cu atât textul scade mai mult
              eroarea (RMSE) pentru acea clădire. În total, textul scade RMSE la {helpfulCount} din{' '}
              {withGain.length} clădiri ({Math.round((helpfulCount / withGain.length) * 100)}%).
            </div>
          )}

          <div className="chart-box" style={{ padding: 4 }}>
            <Plot
              data={traces}
              layout={{
                autosize: true, height: 580, margin: { l: 0, r: 0, t: 8, b: 0 },
                scene: (() => {
                  const ax = (title) => ({
                    title: { text: title, font: { size: 11, color: '#7D7D7D' } },
                    showticklabels: false, showspikes: false,
                    showgrid: true, gridcolor: '#e5e5e5', zeroline: false,
                  })
                  return {
                    xaxis: ax('proiecție MPNet · 1'),
                    yaxis: ax('proiecție MPNet · 2'),
                    zaxis: ax('proiecție MPNet · 3'),
                  }
                })(),
                legend: { orientation: 'h', y: -0.05 },
                font: { family: 'Segoe UI, system-ui, sans-serif', color: '#252525' },
              }}
              config={{ displayModeBar: true, responsive: true, displaylogo: false }}
              style={{ width: '100%' }}
              useResizeHandler
            />
          </div>

          {withGain.length > 0 && (
            <div className="result-section" style={{ marginTop: 22 }}>
              <h2>Unde ajută textul</h2>

              {best && worst && (
                <div className="chart-grid" style={{ marginBottom: 18 }}>
                  <div>
                    <h3>Exemplu unde textul ajută</h3>
                    <div style={{ marginBottom: 6 }}>
                      <span style={{ fontWeight: 600, color: '#252525' }}>{best.building_id}</span>
                      <span style={{ color: 'var(--mid)' }}> · {best.usage || '—'}</span>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      RMSE fără text <b>{best.ntRmse.toFixed(2)}</b>, cu text <b>{best.tRmse.toFixed(2)}</b>{' '}
                      (mai mic cu {Math.abs(best.gain).toFixed(1)}%)
                    </div>
                    <div className="text-box">{best.text_static || '—'}</div>
                  </div>
                  <div>
                    <h3>Exemplu unde textul nu ajută</h3>
                    <div style={{ marginBottom: 6 }}>
                      <span style={{ fontWeight: 600, color: '#252525' }}>{worst.building_id}</span>
                      <span style={{ color: 'var(--mid)' }}> · {worst.usage || '—'}</span>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      RMSE fără text <b>{worst.ntRmse.toFixed(2)}</b>, cu text <b>{worst.tRmse.toFixed(2)}</b>{' '}
                      ({worst.gain > 0 ? 'mai mic' : 'mai mare'} cu {Math.abs(worst.gain).toFixed(1)}%)
                    </div>
                    <div className="text-box">{worst.text_static || '—'}</div>
                  </div>
                </div>
              )}

              <div className="chart-grid">
                <div>
                  <h3>Clădirile unde textul ajută cel mai mult</h3>
                  <table className="data-table">
                    <thead><tr><th>Clădire</th><th>Tip</th><th>Δ RMSE</th></tr></thead>
                    <tbody>
                      {top.map((p) => (
                        <tr key={p.building_id}>
                          <td>{p.building_id}</td>
                          <td>{p.usage || '—'}</td>
                          <td>{pct(p.gain)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h3>Clădirile unde textul ajută cel mai puțin</h3>
                  <table className="data-table">
                    <thead><tr><th>Clădire</th><th>Tip</th><th>Δ RMSE</th></tr></thead>
                    <tbody>
                      {bottom.map((p) => (
                        <tr key={p.building_id}>
                          <td>{p.building_id}</td>
                          <td>{p.usage || '—'}</td>
                          <td>{pct(p.gain)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <h3 style={{ marginTop: 18 }}>Pe tipuri de clădiri</h3>
              <table className="data-table">
                <thead><tr><th>Tip clădire</th><th>Clădiri</th><th>Ajută la</th><th>Δ RMSE mediu</th></tr></thead>
                <tbody>
                  {byType.map((t) => (
                    <tr key={t.usage}>
                      <td>{t.usage}</td>
                      <td>{t.n}</td>
                      <td>{t.help}/{t.n} ({Math.round((t.help / t.n) * 100)}%)</td>
                      <td>{pct(t.mean)}</td>
                    </tr>
                  ))}
                  <tr style={{ fontWeight: 700, borderTop: '2px solid #ccc' }}>
                    <td>Total</td>
                    <td>{withGain.length}</td>
                    <td>{helpfulCount}/{withGain.length} ({Math.round((helpfulCount / withGain.length) * 100)}%)</td>
                    <td>{pct(meanGain)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
