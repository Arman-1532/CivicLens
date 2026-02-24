import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

function Login() {
    const [loginType, setLoginType] = useState('citizen'); // 'citizen' | 'department'
    const [selectedDept, setSelectedDept] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError('');
        try {
            let identifier = email;
            if (loginType === 'department') {
                if (!selectedDept) {
                    setError('Please select a department');
                    setIsLoading(false);
                    return;
                }
                // Match the seeding logic in main.py (Need to replace ALL spaces/slashes)
                identifier = `${selectedDept.toLowerCase().split(' ').join('_').split('/').join('_')}@civiclens.internal`;
            }

            await login(identifier, password);

            // Redirect based on role
            if (loginType === 'department') {
                navigate('/department');
            } else {
                navigate('/dashboard');
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="max-w-md mx-auto mt-10 p-6 bg-white rounded-xl shadow-lg">
            <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">Login to CivicLens</h2>

            {/* Login Type Tabs */}
            <div className="flex mb-6 bg-gray-100 p-1 rounded-lg">
                <button
                    onClick={() => setLoginType('citizen')}
                    className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-all ${loginType === 'citizen' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                >
                    Citizen
                </button>
                <button
                    onClick={() => {
                        setLoginType('department');
                    }}
                    className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-all ${loginType === 'department' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-500 hover:text-gray-700'}`}
                >
                    Department
                </button>
            </div>

            {error && (
                <div className="mb-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg border border-red-200">
                    {error}
                </div>
            )}
            <form onSubmit={handleSubmit} className="space-y-4">
                {loginType === 'citizen' ? (
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                            required
                        />
                    </div>
                ) : (
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Select Department</label>
                        <select
                            value={selectedDept}
                            onChange={(e) => setSelectedDept(e.target.value)}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                            required
                        >
                            <option value="">Choose a Department...</option>
                            <option value="Anti-Corruption Bureau">Anti-Corruption Bureau</option>
                            <option value="Public Utilities Department">Public Utilities Department</option>
                            <option value="Administrative Services">Administrative Services</option>
                            <option value="Women's Commission / HR">Women's Commission / HR</option>
                            <option value="Finance Department">Finance Department</option>
                            <option value="Police Department / Law Enforcement">Police Department / Law Enforcement</option>
                        </select>
                    </div>
                )}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                        required
                    />
                </div>
                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-blue-600 text-white py-2 rounded-lg font-semibold hover:bg-blue-700 transition-colors disabled:bg-gray-400"
                >
                    {isLoading ? 'Verifying...' : 'Access Dashboard'}
                </button>
            </form>
            <p className="mt-6 text-center text-sm text-gray-600">
                Don't have an account?{' '}
                <Link to="/register" className="text-blue-600 font-medium hover:underline">
                    Register here
                </Link>
            </p>
        </div>
    );
}

export default Login;
