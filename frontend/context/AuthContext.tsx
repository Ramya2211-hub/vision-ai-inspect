"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { User } from '@/types';
import { authService } from '@/services/auth';

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (userData: User, token: string) => void;
  logout: () => void;
  refreshUser: () => Promise<User | null>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Initialize auth state from localStorage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;

    try {
      const storedToken = localStorage.getItem('token');
      const storedUser = localStorage.getItem('user');

      if (storedToken) {
        setToken(storedToken);
      }

      if (storedUser) {
        try {
          setUser(JSON.parse(storedUser));
        } catch (e) {
          console.error('[AuthContext] Failed to parse cached user:', e);
          localStorage.removeItem('user');
        }
      }

      // If token exists, optionally verify / refresh user profile in background
      if (storedToken) {
        authService.getMe()
          .then((currentUser) => {
            if (currentUser) {
              setUser(currentUser);
              localStorage.setItem('user', JSON.stringify(currentUser));
            }
          })
          .catch((err) => {
            console.warn('[AuthContext] Session validation warning:', err?.message || err);
            // Only clear if 401 Unauthorized explicitly returned
            if (err?.response?.status === 401) {
              localStorage.removeItem('token');
              localStorage.removeItem('user');
              setToken(null);
              setUser(null);
            }
          })
          .finally(() => {
            setLoading(false);
          });
        return;
      }
    } catch (err) {
      console.error('[AuthContext] Auth initialization error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const login = useCallback((userData: User, newToken: string) => {
    try {
      localStorage.setItem('user', JSON.stringify(userData));
      localStorage.setItem('token', newToken);
    } catch (e) {
      console.error('[AuthContext] Error storing credentials in localStorage:', e);
    }
    setUser(userData);
    setToken(newToken);
  }, []);

  const logout = useCallback(() => {
    try {
      localStorage.removeItem('user');
      localStorage.removeItem('token');
    } catch (e) {
      console.error('[AuthContext] Error clearing credentials:', e);
    }
    setUser(null);
    setToken(null);
    if (typeof window !== 'undefined') {
      window.location.href = '/login';
    }
  }, []);

  const refreshUser = useCallback(async (): Promise<User | null> => {
    try {
      const currentUser = await authService.getMe();
      if (currentUser) {
        setUser(currentUser);
        localStorage.setItem('user', JSON.stringify(currentUser));
        return currentUser;
      }
    } catch (e) {
      console.warn('[AuthContext] Failed to refresh user profile:', e);
    }
    return null;
  }, []);

  const value = {
    user,
    token,
    loading,
    login,
    logout,
    refreshUser,
    isAuthenticated: Boolean(user && token),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    // Fallback if accessed outside provider during testing or isolated render
    if (typeof window !== 'undefined') {
      const storedUser = localStorage.getItem('user');
      const parsed = storedUser ? JSON.parse(storedUser) : null;
      return {
        user: parsed,
        token: localStorage.getItem('token'),
        loading: false,
        login: (u, t) => {
          localStorage.setItem('user', JSON.stringify(u));
          localStorage.setItem('token', t);
        },
        logout: () => {
          localStorage.removeItem('user');
          localStorage.removeItem('token');
          window.location.href = '/login';
        },
        refreshUser: async () => null,
        isAuthenticated: Boolean(parsed),
      };
    }
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
