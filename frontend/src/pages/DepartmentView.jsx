import { useState, useEffect, useCallback } from 'react';
import { getComplaints, updateComplaintStatus } from '../services/api';
import { formatToBDTime } from '../utils/dateUtils';

const DEPARTMENTS = [
    'Anti-Corruption Bureau',
    'Public Utilities Department',
    'Administrative Services',
    "Women's Commission / HR",
    'Finance Department',
    'Police Department / Law Enforcement',
];

const STATUS_LABELS = {
    pending: { label: 'Pending', color: 'bg-yellow-100 text-yellow-800 border-yellow-200' },
    in_progress: { label: 'In Progress', color: 'bg-blue-100   text-blue-800   border-blue-200' },
    resolved: { label: 'Resolved', color: 'bg-green-100  text-green-800  border-green-200' },
};

const URGENCY_COLORS = {
    High: 'text-red-600 bg-red-50',
    Medium: 'text-yellow-600 bg-yellow-50',
    Low: 'text-green-600 bg-green-50',
};

function StatusBadge({ status }) {
    const s = STATUS_LABELS[status] || { label: status, color: 'bg-gray-100 text-gray-800' };
    return (
        <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium border ${s.color}`}>
            {s.label}
        </span>
    );
}

function UpdateModal({ complaint, onClose, onSaved }) {
    const [newStatus, setNewStatus] = useState(complaint.status);
    const [notes, setNotes] = useState(complaint.department_notes || '');
    const [isSaving, setIsSaving] = useState(false);
    const [saveError, setSaveError] = useState('');

    const handleSave = async () => {
        setIsSaving(true);
        setSaveError('');
        try {
            const updated = await updateComplaintStatus(complaint.id, { status: newStatus, notes });
            onSaved(updated);
        } catch (err) {
            setSaveError(err.message || 'Failed to update');
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg">
                <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                    <h3 className="font-semibold text-gray-900">Update Complaint Status</h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-700 transition-colors">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>
                <div className="p-6 space-y-4">
                    {/* Complaint excerpt */}
                    <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1 font-mono">{complaint.tracking_number}</p>
                        <p className="text-sm text-gray-800 line-clamp-3">{complaint.complaint_text}</p>
                    </div>

                    {/* Status selector */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">New Status</label>
                        <div className="flex gap-3">
                            {Object.entries(STATUS_LABELS).map(([val, { label, color }]) => (
                                <button
                                    key={val}
                                    onClick={() => setNewStatus(val)}
                                    className={`flex-1 py-2 rounded-lg text-sm font-medium border transition-all
                    ${newStatus === val ? `${color} ring-2 ring-offset-1 ring-blue-400` : 'bg-white border-gray-200 text-gray-600 hover:border-gray-300'}`}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Notes */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Department Notes <span className="text-gray-400 font-normal">(optional)</span>
                        </label>
                        <textarea
                            value={notes}
                            onChange={(e) => setNotes(e.target.value)}
                            rows={3}
                            placeholder="Add notes about the action taken or expected resolution..."
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                        />
                    </div>

                    {saveError && <p className="text-sm text-red-600">{saveError}</p>}
                </div>
                <div className="px-6 py-4 border-t border-gray-100 flex gap-3 justify-end">
                    <button onClick={onClose} className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors text-sm">
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={isSaving}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm disabled:opacity-50"
                    >
                        {isSaving ? 'Saving...' : 'Save Changes'}
                    </button>
                </div>
            </div>
        </div>
    );
}

function DepartmentView() {
    const [selectedDept, setSelectedDept] = useState(DEPARTMENTS[0]);
    const [statusFilter, setStatusFilter] = useState('all');
    const [complaints, setComplaints] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [modalComplaint, setModalComplaint] = useState(null);

    const load = useCallback(async () => {
        setIsLoading(true);
        setError('');
        try {
            const filters = { department: selectedDept };
            if (statusFilter !== 'all') filters.status = statusFilter;
            const data = await getComplaints(filters);
            setComplaints(data);
        } catch (err) {
            setError(err.message || 'Failed to load complaints');
        } finally {
            setIsLoading(false);
        }
    }, [selectedDept, statusFilter]);

    useEffect(() => { load(); }, [load]);

    const handleSaved = (updated) => {
        setComplaints((prev) => prev.map((c) => (c.id === updated.id ? updated : c)));
        setModalComplaint(null);
    };

    // Stats
    const stats = { pending: 0, in_progress: 0, resolved: 0 };
    complaints.forEach((c) => { if (stats[c.status] !== undefined) stats[c.status]++; });

    return (
        <div className="max-w-6xl mx-auto">
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-gray-900">Department View</h1>
                <p className="text-gray-600 mt-1">View and manage complaints assigned to your department</p>
            </div>

            {/* Department Selector */}
            <div className="bg-white rounded-xl shadow p-5 mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Select Department</label>
                <select
                    value={selectedDept}
                    onChange={(e) => setSelectedDept(e.target.value)}
                    className="w-full md:w-auto min-w-[320px] px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                    {DEPARTMENTS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-white rounded-xl shadow p-5">
                    <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">Pending</p>
                    <p className="text-3xl font-bold text-yellow-600">{stats.pending}</p>
                </div>
                <div className="bg-white rounded-xl shadow p-5">
                    <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">In Progress</p>
                    <p className="text-3xl font-bold text-blue-600">{stats.in_progress}</p>
                </div>
                <div className="bg-white rounded-xl shadow p-5">
                    <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">Resolved</p>
                    <p className="text-3xl font-bold text-green-600">{stats.resolved}</p>
                </div>
            </div>

            {/* Status Filter Tabs */}
            <div className="flex gap-2 mb-4">
                {['all', 'pending', 'in_progress', 'resolved'].map((s) => (
                    <button
                        key={s}
                        onClick={() => setStatusFilter(s)}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
              ${statusFilter === s ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 border border-gray-200 hover:border-gray-300'}`}
                    >
                        {s === 'all' ? 'All' : STATUS_LABELS[s]?.label}
                    </button>
                ))}
                <button onClick={load} className="ml-auto px-3 py-2 text-gray-500 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors text-sm">
                    ↺ Refresh
                </button>
            </div>

            {/* Complaints Table */}
            <div className="bg-white rounded-xl shadow overflow-hidden">
                {isLoading ? (
                    <div className="flex items-center justify-center h-48">
                        <svg className="animate-spin h-8 w-8 text-blue-600" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                    </div>
                ) : error ? (
                    <div className="p-8 text-center">
                        <p className="text-red-600">{error}</p>
                        <button onClick={load} className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm">Retry</button>
                    </div>
                ) : complaints.length === 0 ? (
                    <div className="p-12 text-center">
                        <div className="w-14 h-14 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-3">
                            <svg className="w-7 h-7 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </div>
                        <p className="text-gray-500">No complaints found for this filter</p>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead className="bg-gray-50 text-xs text-gray-500 uppercase tracking-wider">
                                <tr>
                                    <th className="px-5 py-3 text-left">Tracking #</th>
                                    <th className="px-5 py-3 text-left">Citizen</th>
                                    <th className="px-5 py-3 text-left">Location</th>
                                    <th className="px-5 py-3 text-left">Complaint</th>
                                    <th className="px-5 py-3 text-left">Category</th>
                                    <th className="px-5 py-3 text-left">Urgency</th>
                                    <th className="px-5 py-3 text-left">Status</th>
                                    <th className="px-5 py-3 text-left">Date</th>
                                    <th className="px-5 py-3 text-left">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-200">
                                {complaints.map((c) => (
                                    <tr key={c.id} className="hover:bg-gray-50 transition-colors">
                                        <td className="px-5 py-4 font-mono text-xs text-blue-700 whitespace-nowrap">{c.tracking_number}</td>
                                        <td className="px-5 py-4">
                                            <p className="font-medium text-gray-900">{c.citizen_name}</p>
                                            <p className="text-xs text-gray-400">{c.citizen_email}</p>
                                        </td>
                                        <td className="px-5 py-4 text-gray-600 max-w-[120px] truncate" title={c.location}>{c.location}</td>
                                        <td className="px-5 py-4 max-w-[200px]">
                                            <p className="text-gray-800 line-clamp-2" title={c.complaint_text}>{c.complaint_text}</p>
                                            {c.department_notes && (
                                                <p className="text-xs text-blue-600 mt-1 italic">Note: {c.department_notes}</p>
                                            )}
                                        </td>
                                        <td className="px-5 py-4 whitespace-nowrap text-gray-700">{c.category}</td>
                                        <td className="px-5 py-4">
                                            <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${URGENCY_COLORS[c.urgency] || ''}`}>
                                                {c.urgency}
                                            </span>
                                        </td>
                                        <td className="px-5 py-4"><StatusBadge status={c.status} /></td>
                                        <td className="px-5 py-4 text-xs text-gray-400 whitespace-nowrap">
                                            {formatToBDTime(c.created_at)}
                                        </td>
                                        <td className="px-5 py-4">
                                            <button
                                                onClick={() => setModalComplaint(c)}
                                                className="px-3 py-1.5 bg-blue-600 text-white rounded-lg text-xs hover:bg-blue-700 transition-colors font-medium"
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

            {/* Modal */}
            {modalComplaint && (
                <UpdateModal
                    complaint={modalComplaint}
                    onClose={() => setModalComplaint(null)}
                    onSaved={handleSaved}
                />
            )}
        </div>
    );
}

export default DepartmentView;
