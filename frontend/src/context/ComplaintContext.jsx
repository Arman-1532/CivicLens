import { createContext, useContext, useState, useCallback } from 'react';
import { submitComplaint as apiSubmit, getComplaints, updateComplaintStatus as apiUpdateStatus } from '../services/api';

const ComplaintContext = createContext(null);

export function ComplaintProvider({ children }) {
  const [complaints, setComplaints] = useState([]);
  const [currentPrediction, setCurrentPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  /** Submit full complaint form – stores to DB, returns record with tracking number */
  const submitComplaint = useCallback(async (formData) => {
    setIsLoading(true);
    setError(null);
    setCurrentPrediction(null);
    try {
      const result = await apiSubmit(formData);
      setCurrentPrediction(result);
      setComplaints((prev) => [result, ...prev]);
      return result;
    } catch (err) {
      setError(err.message || 'Failed to submit complaint');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /** Fetch complaints from DB (with optional filters) */
  const fetchComplaints = useCallback(async (filters = {}) => {
    try {
      const data = await getComplaints(filters);
      setComplaints(data);
      return data;
    } catch (err) {
      console.error('Failed to fetch complaints:', err);
      return [];
    }
  }, []);

  /** Department updates a complaint's status */
  const updateComplaintStatus = useCallback(async (id, statusData) => {
    try {
      const updated = await apiUpdateStatus(id, statusData);
      setComplaints((prev) =>
        prev.map((c) => (c.id === id ? updated : c))
      );
      return updated;
    } catch (err) {
      console.error('Failed to update status:', err);
      throw err;
    }
  }, []);

  const clearPrediction = useCallback(() => {
    setCurrentPrediction(null);
    setError(null);
  }, []);

  const clearHistory = useCallback(() => {
    setComplaints([]);
    setCurrentPrediction(null);
  }, []);

  return (
    <ComplaintContext.Provider
      value={{
        complaints,
        currentPrediction,
        isLoading,
        error,
        submitComplaint,
        fetchComplaints,
        updateComplaintStatus,
        clearPrediction,
        clearHistory,
      }}
    >
      {children}
    </ComplaintContext.Provider>
  );
}

export function useComplaintContext() {
  const context = useContext(ComplaintContext);
  if (!context) throw new Error('useComplaintContext must be used within a ComplaintProvider');
  return context;
}

export default ComplaintContext;
