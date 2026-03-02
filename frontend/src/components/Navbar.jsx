import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';




function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-40">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 text-blue-600 font-bold text-xl">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            CivicLens
          </Link>

          {/* Nav links */}
          <div className="hidden md:flex items-center gap-1">
            {[
              ...(user?.role !== 'department' ? [
                { to: '/', label: 'Submit Complaint' },
                { to: '/dashboard', label: 'Dashboard' }
              ] : []),
              { to: '/department', label: 'Department View' },
              ...(user?.role === 'admin' ? [{ to: '/admin', label: 'Admin Panel' }] : []),
            ].map(({ to, label }) => {
              const isActive = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
                    ${isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-600 hover:bg-gray-100'}`}
                >
                  {label}
                </Link>
              );
            })}
          </div>

          {/* User Auth Section */}
          <div className="hidden md:flex items-center gap-3 ml-4 pl-4 border-l border-gray-200">
            {user ? (
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-600">
                  Hello, <span className="font-semibold text-gray-800">{user.full_name || user.email}</span>
                </span>
                <button
                  onClick={handleLogout}
                  className="px-4 py-2 text-sm font-semibold text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                >
                  Logout
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Link
                  to="/login"
                  className="px-4 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  Login
                </Link>
                <Link
                  to="/register"
                  className="px-4 py-2 text-sm font-semibold bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors"
                >
                  Register
                </Link>
              </div>
            )}
          </div>


          {/* Mobile menu (simple) */}
          <div className="md:hidden flex items-center gap-2">
            {[
              ...(user?.role !== 'department' ? [
                { to: '/', label: 'Submit' },
                { to: '/dashboard', label: 'Dash' }
              ] : []),
              { to: '/department', label: 'Dept' },
            ].map(({ to, label }) => (
              <Link
                key={to}
                to={to}
                className={`text-xs px-2 py-1 rounded ${location.pathname === to ? 'bg-blue-600 text-white' : 'text-gray-600'}`}
              >
                {label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
