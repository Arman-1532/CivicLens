import { useState, useEffect } from 'react';
import { getComplaints, getStats, updateComplaintStatus, searchComplaintByTracking, getMyComplaints, getMyNotifications, markNotificationRead, getComplaint } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { formatToBDDate } from '../utils/dateUtils';

const STATUS_COLORS = {
  pending: 'bg-yellow-100 text-yellow-800',
  in_progress: 'bg-blue-100 text-blue-800',
  resolved: 'bg-green-100 text-green-800',
};
const STATUS_LABELS = { pending: 'Pending', in_progress: 'In Progress', resolved: 'Resolved' };

const CATEGORY_COLORS = {
  'Corruption': 'bg-red-100 text-red-800',
  'Utility Issue': 'bg-yellow-100 text-yellow-800',
  'Service Delay': 'bg-orange-100 text-orange-800',
  'Harassment': 'bg-purple-100 text-purple-800',
  'Financial Issue': 'bg-green-100 text-green-800',
  'Law Enforcement Issue': 'bg-blue-100 text-blue-800',
};

const URGENCY_COLORS = {
  High: 'text-red-600 font-semibold',
  Medium: 'text-yellow-600 font-semibold',
  Low: 'text-green-600 font-semibold',
};

function Dashboard() {
  const [complaints, setComplaints] = useState([]);
  const [stats, setStats] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ status: '', urgency: '' });

  const { user } = useAuth();

  // Search state
  const [searchTracking, setSearchTracking] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // My Complaints & Notifications
  const [myComplaints, setMyComplaints] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [activeTab, setActiveTab] = useState(user && user.role !== 'department' ? 'my' : 'all'); // Default to 'my' if citizen

  // Detail Modal State
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
  const [detailComplaint, setDetailComplaint] = useState(null);
  const [isDetailLoading, setIsDetailLoading] = useState(false);


  const load = async () => {
    // Only load stats and user-specific data
    // Global complaints list is NOT loaded automatically anymore
    setIsLoading(true);
    setError('');
    try {
      const promises = [getStats()];
      if (user) {
        promises.push(getMyComplaints());
        promises.push(getMyNotifications());
      }

      const [statsData, myData, notifyData] = await Promise.all(promises);
      setStats(statsData);
      if (user) {
        setMyComplaints(myData || []);
        setNotifications(notifyData || []);
      }
    } catch (err) {
      console.error('Initial load failed:', err);
      // We don't necessarily want to show a big error block if global fetch is gone
    } finally {
      setIsLoading(false);
    }
  };


  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchTracking.trim()) return;
    setIsSearching(true);
    setError('');
    setHasSearched(true);
    try {
      const result = await searchComplaintByTracking(searchTracking.trim());
      setComplaints(result ? [result] : []);
      setActiveTab('all');
    } catch (err) {
      setError(err.message);
      setComplaints([]);
    } finally {
      setIsSearching(false);
    }
  };


  const handleMarkRead = async (id) => {
    try {
      await markNotificationRead(id);
      setNotifications(notifications.map(n => n.id === id ? { ...n, is_read: true } : n));
    } catch (err) {
      console.error('Failed to mark read', err);
    }
  };

  const openDetailModal = async (complaintId, notificationId) => {
    setIsDetailLoading(true);
    setIsDetailModalOpen(true);
    try {
      const data = await getComplaint(complaintId);
      setDetailComplaint(data);
      if (notificationId) {
        // Automatically mark as read when viewing details
        await handleMarkRead(notificationId);
      }
    } catch (err) {
      console.error('Failed to fetch complaint details', err);
      setError('Could not load complaint details.');
    } finally {
      setIsDetailLoading(false);
    }
  };

  const closeDetailModal = () => {
    setIsDetailModalOpen(false);
    setDetailComplaint(null);
  };


  useEffect(() => {
    load();
  }, [user]); // Only reload when user auth changes
  const avgConfidence = complaints.length
    ? (complaints.reduce((s, c) => s + c.confidence, 0) / complaints.length * 100).toFixed(1)
    : 0;

  return (
    <div className="max-w-6xl mx-auto relative">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Search for your complaint or view your submitted reports.
          </p>
        </div>
      </div>


      {/* Tracking ID Search */}
      <div className="mb-8">
        <form onSubmit={handleSearch} className="flex gap-2">
          <input
            type="text"
            placeholder="Search by Tracking ID (e.g. CL-2026-00001)..."
            value={searchTracking}
            onChange={(e) => setSearchTracking(e.target.value)}
            className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none shadow-sm"
          />
          <button
            type="submit"
            disabled={isSearching}
            className="px-6 py-2 bg-gray-800 text-white rounded-lg font-semibold hover:bg-gray-900 transition-colors disabled:bg-gray-400"
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>


      {/* Stats Cards - only shown for search results */}
      {activeTab === 'all' && hasSearched && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white rounded-xl shadow p-5">
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Total Found</p>
            <p className="text-3xl font-bold text-gray-900">
              {activeTab === 'all' ? complaints.length : myComplaints.length}
            </p>
          </div>
          <div className="bg-white rounded-xl shadow p-5">
            <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Avg Confidence</p>
            <p className="text-3xl font-bold text-blue-600">{avgConfidence}%</p>
          </div>
        </div>
      )}


      {/* Tabs */}
      {user && user.role !== 'department' && (
        <div className="flex gap-4 mb-6 border-b border-gray-200">
          <button
            onClick={() => setActiveTab('my')}
            className={`pb-2 px-1 text-sm font-medium transition-colors border-b-2 ${activeTab === 'my' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            My Complaints ({myComplaints.length})
          </button>
          <button
            onClick={() => setActiveTab('notifications')}
            className={`pb-2 px-1 text-sm font-medium transition-colors border-b-2 ${activeTab === 'notifications' ? 'border-blue-600 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            Notifications ({notifications.filter(n => !n.is_read).length})
          </button>
        </div>
      )}


      {/* Removed Filters since we only show search results or user data */}


      {/* Table */}
      <div className="bg-white rounded-xl shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-gray-800">
            {activeTab === 'all' ? 'Search Result' : 'My Complaints'}
            {(activeTab === 'all' ? complaints.length : myComplaints.length) > 0 && ` (${activeTab === 'all' ? complaints.length : myComplaints.length})`}
          </h2>
          {activeTab === 'all' && hasSearched && (
            <button
              onClick={() => { setComplaints([]); setHasSearched(false); if (user) setActiveTab('my'); }}
              className="text-xs text-blue-600 hover:underline"
            >
              Clear Search
            </button>
          )}
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center h-48">
            <p>Loading...</p>
          </div>
        ) : error ? (
          <div className="p-8 text-center">
            <p className="text-red-600 mb-3">{error}</p>
            <button onClick={load} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm">Retry</button>
          </div>
        ) : activeTab === 'notifications' ? (
          <div className="divide-y divide-gray-100">
            {notifications.length === 0 ? (
              <p className="p-8 text-center text-gray-500">No notifications yet</p>
            ) : (
              notifications.map((n) => (
                <div key={n.id} className={`p-4 flex justify-between items-center ${n.is_read ? 'bg-white' : 'bg-blue-50'}`}>
                  <div>
                    <p className="text-sm text-gray-800">{n.message}</p>
                    <p className="text-xs text-gray-400 mt-1">{formatToBDDate(n.created_at)}</p>
                  </div>
                  {!n.is_read ? (
                    <div className="flex gap-2">
                      <button
                        onClick={() => openDetailModal(n.complaint_id, n.id)}
                        className="text-xs text-blue-600 font-semibold hover:underline"
                      >
                        View Details
                      </button>
                      <button
                        onClick={() => handleMarkRead(n.id)}
                        className="text-xs text-gray-500 hover:text-gray-700"
                      >
                        Mark as Read
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => openDetailModal(n.complaint_id)}
                      className="text-xs text-blue-600 hover:underline"
                    >
                      View Details
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        ) : (activeTab === 'all' ? complaints : myComplaints).length === 0 ? (
          <div className="p-12 text-center">
            <p className="text-gray-500">No complaints found</p>
            <p className="text-sm text-gray-400 mt-1">Submit a complaint to see it here</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wider">
                <tr>
                  <th className="px-5 py-3 text-left">Tracking #</th>
                  <th className="px-5 py-3 text-left">Citizen</th>
                  <th className="px-5 py-3 text-left">Complaint</th>
                  <th className="px-5 py-3 text-left">Dept & Category</th>
                  <th className="px-5 py-3 text-left">Urgency</th>
                  <th className="px-5 py-3 text-left">Status</th>
                  <th className="px-5 py-3 text-left">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {(activeTab === 'all' ? complaints : myComplaints).map((c) => (
                  <tr key={c.id} className="hover:bg-gray-50 transition-colors">
                    <td className="px-5 py-4 font-mono text-xs text-blue-700 whitespace-nowrap align-top">{c.tracking_number}</td>
                    <td className="px-5 py-4 align-top">
                      <p className="font-medium text-gray-900">{c.citizen_name}</p>
                      <p className="text-xs text-gray-400">{c.location}</p>
                    </td>
                    <td className="px-5 py-4 max-w-[200px] align-top">
                      <p className="text-gray-800 line-clamp-2 mb-1" title={c.complaint_text}>{c.complaint_text}</p>
                      {c.department_notes && (
                        <p className="text-xs text-gray-500 italic border-l-2 border-blue-200 pl-2">
                          Note: {c.department_notes}
                        </p>
                      )}
                    </td>
                    <td className="px-5 py-4 align-top">
                      <p className="text-xs font-semibold text-gray-700 mb-1 truncate max-w-[140px]" title={c.department}>{c.department}</p>
                      <span className={`inline-flex px-2 py-0.5 text-[10px] font-medium rounded-full ${CATEGORY_COLORS[c.category] || 'bg-gray-100'}`}>
                        {c.category}
                      </span>
                    </td>
                    <td className={`px-5 py-4 text-xs align-top ${URGENCY_COLORS[c.urgency] || ''}`}>{c.urgency}</td>
                    <td className="px-5 py-4 align-top">
                      <span className={`inline-flex px-2 py-0.5 text-xs font-medium rounded-full ${STATUS_COLORS[c.status] || 'bg-gray-100 text-gray-800'}`}>
                        {STATUS_LABELS[c.status] || c.status}
                      </span>
                    </td>
                    <td className="px-5 py-4 align-top">
                      <span className="text-gray-400 text-xs italic">View Only</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {isDetailModalOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="font-bold text-gray-900">
                {isDetailLoading ? 'Loading Details...' : `Complaint: ${detailComplaint?.tracking_number}`}
              </h3>
              <button onClick={closeDetailModal} className="text-gray-400 hover:text-gray-700 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6">
              {isDetailLoading ? (
                <div className="flex flex-col items-center justify-center py-10 space-y-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                </div>
              ) : detailComplaint ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <p className="text-gray-500 text-xs uppercase mb-1">Status</p>
                      <span className={`inline-flex px-2 py-0.5 text-xs font-medium rounded-full ${STATUS_COLORS[detailComplaint.status] || 'bg-gray-100 text-gray-800'}`}>
                        {STATUS_LABELS[detailComplaint.status] || detailComplaint.status}
                      </span>
                    </div>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <p className="text-gray-500 text-xs uppercase mb-1">Department</p>
                      <p className="font-semibold text-gray-800">{detailComplaint.department}</p>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-gray-900 mb-2">Complaint Text</h4>
                    <p className="text-sm text-gray-600 bg-gray-50 p-4 rounded-xl border border-gray-100 italic">
                      "{detailComplaint.complaint_text}"
                    </p>
                  </div>

                  {detailComplaint.department_notes && (
                    <div className="bg-blue-50 border-l-4 border-blue-500 p-4 rounded-r-xl">
                      <h4 className="text-sm font-bold text-blue-900 mb-1">Department Response</h4>
                      <p className="text-sm text-blue-800">
                        {detailComplaint.department_notes}
                      </p>
                    </div>
                  )}

                  <div className="flex justify-between items-center text-xs text-gray-400 pt-4 border-t border-gray-100">
                    <span>Submitted on: {formatToBDDate(detailComplaint.created_at)}</span>
                    <span>Last updated: {formatToBDDate(detailComplaint.updated_at)}</span>
                  </div>
                </div>
              ) : (
                <p className="text-center text-red-600">Failed to load content.</p>
              )}
            </div>

            <div className="px-6 py-4 bg-gray-50 border-t border-gray-100 flex justify-end">
              <button
                onClick={closeDetailModal}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-semibold shadow-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
