import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import SideNav from './components/SideNav'
import Home from './pages/Home'
import Building from './pages/Building'
import Inference from './pages/Inference'
import Embeddings from './pages/Embeddings'
import Archive from './pages/Archive'

export default function App() {
  return (
    <div className="app">
      <Header />
      <div className="shell">
        <SideNav />
        <main className="content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/cladire" element={<Building />} />
            <Route path="/inferenta" element={<Inference />} />
            <Route path="/embeddings" element={<Embeddings />} />
            <Route path="/arhiva" element={<Archive />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
