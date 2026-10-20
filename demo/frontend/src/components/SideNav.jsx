import { NavLink } from 'react-router-dom'

const SECTIONS = [
  { to: '/cladire',    title: 'Clădire',    desc: 'Profil, istoric și metrici pentru o clădire.', accent: '#4a6fa5' },
  { to: '/inferenta',  title: 'Inferență',  desc: 'Rulează modelele pe o zi și compară cu realul.', accent: '#b5654a' },
  { to: '/embeddings', title: 'Embeddings', desc: 'Spațiul semantic al descrierilor de clădiri.', accent: '#9a7bb0' },
  { to: '/arhiva',     title: 'Arhivă',     desc: 'Rulările de inferență salvate, reafișabile.', accent: '#b08a3e' },
]

export default function SideNav() {
  return (
    <aside className="side-nav">
      <NavLink to="/" end className="side-home">Acasă</NavLink>
      <div className="side-title">Secțiuni</div>
      <nav className="side-list">
        {SECTIONS.map((c) => (
          <NavLink
            key={c.to}
            to={c.to}
            className="nav-card side-card"
            style={{ '--accent': c.accent }}
          >
            <span className="nav-title">{c.title}<span className="nav-arrow">→</span></span>
            <span className="nav-desc">{c.desc}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
