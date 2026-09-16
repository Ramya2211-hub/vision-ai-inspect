"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { authService } from '@/services/auth';
import { useAuth } from '@/hooks/useAuth';
import { formatApiError } from '@/services/api';
import Link from 'next/link';
import { AlertCircle, CheckCircle2, ShieldCheck, ArrowRight, Lock, User as UserIcon, Loader2 } from 'lucide-react';

interface LoginFormValues {
  credential: string;
  password: string;
}

export default function LoginPage() {
  const [error, setError] = useState('');
  const [infoMessage, setInfoMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
  const router = useRouter();
  const { login, isAuthenticated, loading: authLoading } = useAuth();
  const { register, handleSubmit, formState: { errors } } = useForm<LoginFormValues>();

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const searchParams = new URLSearchParams(window.location.search);
      if (searchParams.get('session_expired') === 'true') {
        setInfoMessage('Your session has expired. Please sign in to continue.');
      } else if (searchParams.get('registered') === 'true') {
        setInfoMessage('Account registered successfully in database! Please sign in.');
      }
    }
  }, []);

  // If already logged in, redirect straight to dashboard
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      router.replace('/dashboard');
    }
  }, [isAuthenticated, authLoading, router]);

  const onSubmit = async (data: LoginFormValues) => {
    try {
      setIsLoading(true);
      setError('');
      setInfoMessage('');

      // Show warming notice if cloud backend takes > 2.5s (Render cold start)
      const warmTimer = setTimeout(() => {
        setIsWarmingUp(true);
      }, 2500);

      const credential = data.credential.trim();
      const password = data.password;

      // 1. Authenticate with backend API
      const res = await authService.login(credential, password);
      clearTimeout(warmTimer);
      setIsWarmingUp(false);

      if (!res?.access_token) {
        throw new Error('No access token received from authentication server.');
      }

      // Store token immediately
      localStorage.setItem('token', res.access_token);

      // 2. Fetch authenticated user profile
      const userProfile = await authService.getMe();
      if (!userProfile) {
        throw new Error('Failed to retrieve user profile after authentication.');
      }

      // 3. Update global AuthContext state
      login(userProfile, res.access_token);

      // 4. Navigate to dashboard
      router.replace('/dashboard');
    } catch (err: any) {
      console.error('[LoginPage] Sign in error:', err);
      setIsWarmingUp(false);
      const formatted = formatApiError(err, 'Sign in failed. Please verify your username or password.');
      setError(formatted);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-2xl shadow-2xl w-full max-w-md border border-slate-100">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-3">
            <div className="w-14 h-14 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center shadow-inner">
              <ShieldCheck size={32} />
            </div>
          </div>
          <h1 className="text-2xl font-black tracking-tight text-slate-900">VISIONINSPECT AI</h1>
          <p className="text-slate-500 font-medium text-xs mt-1">Industrial Quality Inspection & Defect Analysis</p>
        </div>

        {infoMessage && (
          <div className="mb-5 p-3.5 bg-blue-50 border border-blue-200 text-blue-800 rounded-xl text-xs flex items-center gap-2.5 animate-fadeIn">
            <CheckCircle2 size={18} className="text-blue-600 shrink-0" />
            <span className="font-medium">{infoMessage}</span>
          </div>
        )}

        {error && (
          <div className="mb-5 p-3.5 bg-red-50 text-red-700 rounded-xl text-xs font-medium border border-red-200 flex items-start gap-2.5 animate-fadeIn">
            <AlertCircle size={18} className="text-red-600 shrink-0 mt-0.5" />
            <span className="flex-1 leading-relaxed">{error}</span>
          </div>
        )}

        {isWarmingUp && (
          <div className="mb-5 p-3.5 bg-amber-50 text-amber-800 rounded-xl text-xs font-medium border border-amber-200 flex items-center gap-2.5">
            <Loader2 size={16} className="text-amber-600 animate-spin shrink-0" />
            <span>Connecting to cloud inspection server (Render cold-start)... please wait a moment.</span>
          </div>
        )}

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Username or Email
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <UserIcon size={16} />
              </div>
              <input 
                {...register("credential", { 
                  required: "Username or email is required",
                  minLength: { value: 2, message: "Must be at least 2 characters" }
                })}
                type="text" 
                autoComplete="username"
                disabled={isLoading}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm transition-all text-slate-900 placeholder:text-slate-400 disabled:bg-slate-50"
                placeholder="admin / quality_eng"
              />
            </div>
            {errors.credential && (
              <p className="text-red-600 text-xs mt-1 font-medium">{errors.credential.message}</p>
            )}
          </div>
          
          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <Lock size={16} />
              </div>
              <input 
                {...register("password", { 
                  required: "Password is required" 
                })}
                type="password" 
                autoComplete="current-password"
                disabled={isLoading}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm transition-all text-slate-900 placeholder:text-slate-400 disabled:bg-slate-50"
                placeholder="••••••••"
              />
            </div>
            {errors.password && (
              <p className="text-red-600 text-xs mt-1 font-medium">{errors.password.message}</p>
            )}
          </div>

          <button 
            type="submit" 
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-bold py-3 px-4 rounded-xl transition-all disabled:opacity-70 disabled:cursor-not-allowed flex justify-center items-center h-12 text-sm shadow-md hover:shadow-lg cursor-pointer mt-2 group"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <Loader2 size={18} className="animate-spin" />
                <span>Signing in...</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span>Sign In to Platform</span>
                <ArrowRight size={16} className="group-hover:translate-x-0.5 transition-transform" />
              </div>
            )}
          </button>
        </form>
        
        <div className="mt-6 pt-5 border-t border-slate-100 text-center">
          <p className="text-xs text-slate-600 mb-2 font-medium">New operator or quality engineer?</p>
          <Link 
            href="/register"
            className="inline-flex items-center gap-1.5 text-sm text-blue-600 font-bold hover:text-blue-700 hover:underline cursor-pointer"
          >
            <span>Register a new database account</span>
            <ArrowRight size={14} />
          </Link>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-100 text-center">
          <p className="text-[11px] text-slate-400">VisionInspect AI • Secure Industrial Authentication</p>
        </div>
      </div>
    </div>
  );
}
