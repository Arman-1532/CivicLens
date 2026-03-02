import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { loginUser, registerUser, getUserMe } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    const login = async (email, password) => {
        const data = await loginUser(email, password);
        localStorage.setItem('token', data.access_token);
        // Fetch actual user profile
        try {
            const profile = await getUserMe();
            setUser(profile);
        } catch (err) {
            console.error('Failed to fetch profile after login', err);
            setUser({ email }); // Fallback
        }
        return data;
    };

    const register = async (userData) => {
        return await registerUser(userData);
    };

    const logout = useCallback(() => {
        localStorage.removeItem('token');
        setUser(null);
    }, []);

    useEffect(() => {
        const initAuth = async () => {
            const token = localStorage.getItem('token');
            if (token) {
                try {
                    const profile = await getUserMe();
                    setUser(profile);
                } catch (err) {
                    console.error('Failed to restore session', err);
                    localStorage.removeItem('token');
                }
            }
            setLoading(false);
        };
        initAuth();
    }, []);

    return (
        <AuthContext.Provider value={{ user, loading, login, register, logout }}>
            {children}
        </AuthContext.Provider>
    );
}


export function useAuth() {
    const context = useContext(AuthContext);
    if (!context) throw new Error('useAuth must be used within an AuthProvider');
    return context;
}
