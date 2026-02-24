import { useState } from 'react';
import ComplaintForm from '../components/ComplaintForm';
import ResultCard from '../components/ResultCard';
import { useComplaintContext } from '../context/ComplaintContext';

const CATEGORIES = [
  { name: 'Corruption', icon: '🏛️', color: 'bg-red-50    border-red-200' },
  { name: 'Utility Issue', icon: '💡', color: 'bg-yellow-50  border-yellow-200' },
  { name: 'Service Delay', icon: '⏳', color: 'bg-orange-50  border-orange-200' },
  { name: 'Harassment', icon: '⚠️', color: 'bg-purple-50  border-purple-200' },
  { name: 'Financial Issue', icon: '💰', color: 'bg-green-50   border-green-200' },
  { name: 'Law Enforcement', icon: '🚔', color: 'bg-blue-50    border-blue-200' },
];

function Home() {
  const { submitComplaint, currentPrediction, isLoading, error, clearPrediction } = useComplaintContext();
  const [showResult, setShowResult] = useState(false);

  const handleSubmit = async (formData) => {
    try {
      await submitComplaint(formData);
      setShowResult(true);
    } catch (err) {
      console.error('Submission failed:', err);
    }
  };

  const handleNewComplaint = () => {
    clearPrediction();
    setShowResult(false);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Hero */}
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          AI-Powered Complaint Submission
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Submit your complaint and our AI will automatically classify it,
          assign a tracking number, and route it to the appropriate government department.
        </p>
      </div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-8">
        {/* Form */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-5 flex items-center gap-2">
            <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
            Submit a Complaint
          </h2>

          <ComplaintForm onSubmit={handleSubmit} isLoading={isLoading} />

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-700 text-sm flex items-center gap-2">
                <svg className="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </p>
            </div>
          )}
        </div>

        {/* Result / How it works */}
        <div>
          {showResult && currentPrediction ? (
            <ResultCard prediction={currentPrediction} onNewComplaint={handleNewComplaint} />
          ) : (
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 h-full flex flex-col justify-center">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-800 mb-4">How It Works</h3>
                <ol className="text-sm text-gray-600 space-y-3 text-left max-w-xs mx-auto">
                  {[
                    'Fill in your details and complaint',
                    'Our AI analyses and classifies the complaint',
                    'A unique tracking number is generated',
                    'Complaint is routed to the responsible department',
                    'Department reviews and resolves your complaint',
                  ].map((step, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-600 text-white text-xs flex items-center justify-center font-bold mt-0.5">
                        {i + 1}
                      </span>
                      {step}
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Categories */}
      <div className="mt-12">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Complaint Categories Handled</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {CATEGORIES.map((cat) => (
            <div key={cat.name} className={`${cat.color} border rounded-lg p-4 text-center hover:shadow-md transition-shadow`}>
              <span className="text-2xl mb-2 block">{cat.icon}</span>
              <span className="text-sm font-medium text-gray-700">{cat.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Home;
