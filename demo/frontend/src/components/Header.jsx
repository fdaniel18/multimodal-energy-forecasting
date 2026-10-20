import { NavLink } from 'react-router-dom'

export default function Header() {
  return (
    <header className="app-header">
      <NavLink to="/" className="brand">Predicții Consum Energetic</NavLink>
    </header>
  )
}
