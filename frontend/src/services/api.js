import axios from 'axios';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
});

api.interceptors.request.use(
  (config) => { console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`); return config; },
  (error) => { console.error('API Request Error:', error); return Promise.reject(error); }
);

api.interceptors.response.use(
  (response) => response,
  (error) => { console.error('API Response Error:', error.response?.data || error.message); return Promise.reject(error); }
);

/** Submit a full complaint (with citizen info) – stored in DB */
export const submitComplaint = async (formData) => {
  try {
    const response = await api.post('/api/v1/complaints/submit', formData);
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Submission failed');
    throw new Error('Network error. Please check your connection.');
  }
};

/** List complaints – supports optional { department, status, urgency } filters */
export const getComplaints = async (filters = {}) => {
  try {
    const params = {};
    if (filters.department) params.department = filters.department;
    if (filters.status)     params.status     = filters.status;
    if (filters.urgency)    params.urgency    = filters.urgency;
    const response = await api.get('/api/v1/complaints', { params });
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Failed to fetch complaints');
    throw new Error('Network error. Please check your connection.');
  }
};

/** Get a single complaint by ID */
export const getComplaint = async (id) => {
  try {
    const response = await api.get(`/api/v1/complaints/${id}`);
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Failed to fetch complaint');
    throw new Error('Network error.');
  }
};

/** Department updates complaint status */
export const updateComplaintStatus = async (id, { status, notes }) => {
  try {
    const response = await api.patch(`/api/v1/complaints/${id}/status`, { status, notes });
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Failed to update status');
    throw new Error('Network error.');
  }
};

/** Dashboard statistics */
export const getStats = async () => {
  try {
    const response = await api.get('/api/v1/complaints/stats');
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Failed to fetch stats');
    throw new Error('Network error.');
  }
};

/** Get all categories */
export const getCategories = async () => {
  try {
    const response = await api.get('/api/v1/complaints/categories');
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Failed to fetch categories');
    throw new Error('Network error.');
  }
};

/** Check API health */
export const checkHealth = async () => {
  try {
    const response = await api.get('/api/v1/health');
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Health check failed');
    throw new Error('API is not available');
  }
};

/** Get model info */
export const getModelInfo = async () => {
  try {
    const response = await api.get('/api/v1/complaints/model-info');
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Failed to fetch model info');
    throw new Error('Network error.');
  }
};

// Legacy alias
export const classifyComplaint = async (text) => {
  try {
    const response = await api.post('/api/v1/complaints/predict', { text });
    return response.data;
  } catch (error) {
    if (error.response) throw new Error(error.response.data.detail || 'Classification failed');
    throw new Error('Network error.');
  }
};

export default api;
