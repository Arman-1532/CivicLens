import { useState, useEffect } from 'react';
import { checkHealth, getModelInfo, getCategories } from '../services/api';

function AdminPanel() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [categories, setCategories] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [healthData, modelData, categoriesData] = await Promise.all([
        checkHealth(),
        getModelInfo(),
        getCategories(),
      ]);

      setHealth(healthData);
      setModelInfo(modelData);
      setCategories(categoriesData.categories || []);
    } catch (err) {
      setError(err.message || 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  };

  const StatusBadge = ({ status }) => {
    const isHealthy = status === 'healthy' || status === true;
    return (
      <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
        isHealthy ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
      }`}>
        <span className={`w-2 h-2 rounded-full mr-2 ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`}></span>
        {isHealthy ? 'Healthy' : 'Degraded'}
      </span>
    );
  };

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <svg className="animate-spin h-10 w-10 text-blue-600 mx-auto mb-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <p className="text-gray-600">Loading system information...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
          <svg className="w-12 h-12 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <h3 className="text-lg font-semibold text-red-800 mb-2">Connection Error</h3>
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={fetchAllData}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Admin Panel</h1>
          <p className="text-gray-600 mt-1">System status and configuration</p>
        </div>
        <button
          onClick={fetchAllData}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* System Health */}
      <div className="bg-white rounded-xl shadow mb-6">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800">System Health</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-500 mb-1">API Status</p>
              <StatusBadge status={health?.status} />
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Model Loaded</p>
              <StatusBadge status={health?.model_loaded} />
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Vectorizer Loaded</p>
              <StatusBadge status={health?.vectorizer_loaded} />
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Version</p>
              <span className="text-lg font-semibold text-gray-900">{health?.version || 'N/A'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Information */}
      <div className="bg-white rounded-xl shadow mb-6">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800">Model Information</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
            <div>
              <p className="text-sm text-gray-500 mb-1">Model Type</p>
              <p className="text-lg font-semibold text-gray-900">{modelInfo?.model_type || 'SVC'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Vectorizer Type</p>
              <p className="text-lg font-semibold text-gray-900">{modelInfo?.vectorizer_type || 'TfidfVectorizer'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Vocabulary Size</p>
              <p className="text-lg font-semibold text-gray-900">
                {modelInfo?.vocabulary_size?.toLocaleString() || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 mb-1">Ready Status</p>
              <StatusBadge status={modelInfo?.ready} />
            </div>
            <div className="col-span-2">
              <p className="text-sm text-gray-500 mb-1">Categories Supported</p>
              <p className="text-lg font-semibold text-gray-900">{modelInfo?.categories?.length || 6}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Categories */}
      <div className="bg-white rounded-xl shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800">Supported Categories</h2>
        </div>
        <div className="p-6">
          <div className="grid gap-3">
            {categories.map((cat, index) => (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{cat.name}</p>
                  <p className="text-sm text-gray-500">{cat.department}</p>
                </div>
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  Active
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* API Endpoints */}
      <div className="bg-white rounded-xl shadow mt-6">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800">API Endpoints</h2>
        </div>
        <div className="p-6">
          <div className="space-y-3 font-mono text-sm">
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-bold">GET</span>
              <span className="text-gray-700">/api/v1/health</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-bold">POST</span>
              <span className="text-gray-700">/api/v1/complaints/predict</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-bold">GET</span>
              <span className="text-gray-700">/api/v1/complaints/categories</span>
            </div>
            <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-bold">GET</span>
              <span className="text-gray-700">/api/v1/complaints/model-info</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdminPanel;

