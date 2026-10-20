export const MODEL_COLORS = {
  real: '#252525',
  perhouse_notext: '#8e7cc3',
  perhouse_text: '#f2a13a',
  global_notext: '#2f6df0',
  global_text: '#e2483a',
  patchtst: '#18a558',
}

export const MODEL_LABELS = {
  real: 'Real',
  perhouse_notext: 'BiLSTM per clădire',
  perhouse_text: 'Hibrid per clădire',
  global_notext: 'BiLSTM global',
  global_text: 'Hibrid global',
  patchtst: 'PatchTST',
}

export const MODEL_ORDER = [
  'perhouse_notext', 'perhouse_text', 'global_notext', 'global_text', 'patchtst',
]

export const CATEGORICAL = [
  '#2f6df0', '#e2483a', '#18a558', '#f2a13a', '#8e44ad',
  '#16a2b8', '#d63384', '#0a7d6b', '#fd7e14', '#6f42c1',
]

export const num3 = (v) =>
  (typeof v === 'number' && Number.isFinite(v) ? +v.toFixed(3) : v)
