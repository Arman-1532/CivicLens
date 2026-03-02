import { useState, useCallback } from 'react';
import { classifyComplaint } from '../services/api';

/**
 * Custom hook for complaint classification
 */
export function useComplaint() {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const classify = useCallback(async (text) => {
    if (!text || text.trim().length < 10) {
      setError('Please enter at least 10 characters');
      return null;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await classifyComplaint(text);
      setPrediction(result);
      return result;
    } catch (err) {
      setError(err.message || 'Classification failed');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setPrediction(null);
    setError(null);
  }, []);

  return {
    prediction,
    isLoading,
    error,
    classify,
    reset,
  };
}

export default useComplaint;

