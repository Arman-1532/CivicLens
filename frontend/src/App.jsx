import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ComplaintProvider } from './context/ComplaintContext'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import AdminPanel from './pages/AdminPanel'
import DepartmentView from './pages/DepartmentView'

function App() {
  return (
    <ComplaintProvider>
      <Router>
        <div className="min-h-screen bg-gray-50 flex flex-col">
          <Navbar />
          <main className="flex-1 container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/department" element={<DepartmentView />} />
              <Route path="/admin" element={<AdminPanel />} />
            </Routes>
          </main>
          <footer className="bg-white border-t border-gray-200 py-5 mt-auto">
            <div className="container mx-auto px-4 text-center text-gray-500 text-sm">
              <p>© 2026 CivicLens — AI-Powered Complaint Submission &amp; Management System</p>
              <p className="text-xs mt-1 text-gray-400">Powered by Machine Learning for Better Governance</p>
            </div>
          </footer>
        </div>
      </Router>
    </ComplaintProvider>
  )
}

export default App
