import { useState, useEffect } from 'react';
import { getComplaints, getStats, updateComplaintStatus } from '../services/api';

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
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ status: '', urgency: '' });

  // Update Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [newStatus, setNewStatus] = useState('');
  const [departmentNotes, setDepartmentNotes] = useState('');
  const [isUpdating, setIsUpdating] = useState(false);

  const load = async () => {
    setIsLoading(true);
    setError('');
    try {
      const f = {};
      if (filters.status) f.status = filters.status;
      if (filters.urgency) f.urgency = filters.urgency;
      const [data, statsData] = await Promise.all([getComplaints(f), getStats()]);
      setComplaints(data);
      setStats(statsData);
    } catch (err) {
      setError(err.message || 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => { load(); }, [filters]); // eslint-disable-line

  const openUpdateModal = (complaint) => {
    setSelectedComplaint(complaint);
    setNewStatus(complaint.status);
    setDepartmentNotes(complaint.department_notes || '');
    setIsModalOpen(true);
  };

  const closeUpdateModal = () => {
    setIsModalOpen(false);
    setSelectedComplaint(null);
    setNewStatus('');
    setDepartmentNotes('');
  };

  const handleUpdateStatus = async () => {
    if (!selectedComplaint) return;
    setIsUpdating(true);
    try {
      await updateComplaintStatus(selectedComplaint.id, {
        status: newStatus,
        notes: departmentNotes
      });
      // Refresh data
      await load();
      closeUpdateModal();
    } catch (err) {
      alert('Failed to update status: ' + err.message);
    } finally {
      setIsUpdating(false);
    }
  };

  const avgConfidence = complaints.length
    ? (complaints.reduce((s, c) => s + c.confidence, 0) / complaints.length * 100).toFixed(1)
    : 0;

  return (
    <div className="max-w-6xl mx-auto relative">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">All complaints stored in the database</p>
        </div>
        <button
          onClick={load}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2 text-sm"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-xl shadow p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Total</p>
          <p className="text-3xl font-bold text-gray-900">{stats?.total ?? '—'}</p>
        </div>
        <div className="bg-white rounded-xl shadow p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Pending</p>
          <p className="text-3xl font-bold text-yellow-600">{stats?.by_status?.pending ?? '—'}</p>
        </div>
        <div className="bg-white rounded-xl shadow p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">High Urgency</p>
          <p className="text-3xl font-bold text-red-600">{stats?.by_urgency?.High ?? '—'}</p>
        </div>
        <div className="bg-white rounded-xl shadow p-5">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">Avg Confidence</p>
          <p className="text-3xl font-bold text-blue-600">{avgConfidence}%</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-3 mb-4 flex-wrap">
        <select
          value={filters.status}
          onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value }))}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All Statuses</option>
          <option value="pending">Pending</option>
          <option value="in_progress">In Progress</option>
          <option value="resolved">Resolved</option>
        </select>
        <select
          value={filters.urgency}
          onChange={(e) => setFilters((f) => ({ ...f, urgency: e.target.value }))}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All Urgencies</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800">
            Complaints ({complaints.length})
          </h2>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center h-48">
            <svg className="animate-spin h-8 w-8 text-blue-500" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
        ) : error ? (
          <div className="p-8 text-center">
            <p className="text-red-600 mb-3">{error}</p>
            <button onClick={load} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm">Retry</button>
          </div>
        ) : complaints.length === 0 ? (
          <div className="p-12 text-center">
            <p className="text-gray-500">No complaints found</p>
            <p className="text-sm text-gray-400 mt-1">Submit a complaint on the Home page to see it here</p>
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
                {complaints.map((c) => (
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
                      <button
                        onClick={() => openUpdateModal(c)}
                        className="text-blue-600 hover:text-blue-900 text-xs font-medium bg-blue-50 px-3 py-1.5 rounded hover:bg-blue-100 transition-colors"
                      >
                        Update
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Update Modal */}
      {isModalOpen && selectedComplaint && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-xl max-w-lg w-full overflow-hidden">
            <div className="bg-gray-50 px-6 py-4 border-b border-gray-200 flex justify-between items-center">
              <h3 className="text-lg font-bold text-gray-800">
                Update Status: {selectedComplaint.tracking_number}
              </h3>
              <button onClick={closeUpdateModal} className="text-gray-400 hover:text-gray-600">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 space-y-4">
              {/* Current Info */}
              <div className="bg-blue-50 p-3 rounded-lg text-sm mb-4">
                <p><span className="font-semibold">Categories:</span> {selectedComplaint.category}</p>
                <p><span className="font-semibold">Department:</span> {selectedComplaint.department}</p>
                <p className="mt-1"><span className="font-semibold">Complaint:</span> <span className="text-gray-600">{selectedComplaint.complaint_text}</span></p>
              </div>

              {/* Status Update */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                <div className="flex gap-4">
                  {Object.keys(STATUS_LABELS).map((statusKey) => (
                    <label key={statusKey} className="flex items-center">
                      <input
                        type="radio"
                        name="status"
                        value={statusKey}
                        checked={newStatus === statusKey}
                        onChange={(e) => setNewStatus(e.target.value)}
                        className="w-4 h-4 text-blue-600 border-gray-300 focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">{STATUS_LABELS[statusKey]}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Notes */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Department Notes</label>
                <textarea
                  value={departmentNotes}
                  onChange={(e) => setDepartmentNotes(e.target.value)}
                  placeholder="Add notes about actions taken..."
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            <div className="bg-gray-50 px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
              <button
                onClick={closeUpdateModal}
                disabled={isUpdating}
                className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleUpdateStatus}
                disabled={isUpdating}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center gap-2"
              >
                {isUpdating ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
