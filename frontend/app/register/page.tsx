"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { authService } from '@/services/auth';
import { useAuth } from '@/hooks/useAuth';
import { formatApiError } from '@/services/api';
import Link from 'next/link';
import { AlertCircle, ShieldCheck, ArrowRight, User, Mail, Lock, Briefcase, Loader2 } from 'lucide-react';

interface RegisterFormValues {
  username: string;
  email: string;
  password: string;
  role_name: string;
}

export default function RegisterPage() {
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
  const router = useRouter();
  const { login } = useAuth();
  const { register, handleSubmit, formState: { errors } } = useForm<RegisterFormValues>({
    defaultValues: {
      role_name: 'QUALITY_ENGINEER'
    }
  });

  const onSubmit = async (data: RegisterFormValues) => {
    try {
      setIsLoading(true);
      setError('');

      const warmTimer = setTimeout(() => {
        setIsWarmingUp(true);
      }, 2500);

      const username = data.username.trim();
      const email = data.email.trim();
      const password = data.password;
      const role_name = data.role_name;

      // 1. Create user in the database
      await authService.register({
        username,
        email,
        password,
        role_name
      });

      // 2. Auto-login immediately after successful registration
      const loginRes = await authService.login(username, password);
      clearTimeout(warmTimer);
      setIsWarmingUp(false);

      if (!loginRes?.access_token) {
        throw new Error('User registered in database, but login token was not returned.');
      }

      localStorage.setItem('token', loginRes.access_token);

      // 3. Fetch user profile
      const userProfile = await authService.getMe();
      if (!userProfile) {
        throw new Error('Failed to retrieve user profile after registration.');
      }

      // 4. Update AuthContext globally
      login(userProfile, loginRes.access_token);

      // 5. Navigate to dashboard
      router.replace('/dashboard');
    } catch (err: any) {
      console.error('[RegisterPage] Registration error:', err);
      setIsWarmingUp(false);
      const formatted = formatApiError(err, 'Registration failed. Please verify user details.');
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
          <p className="text-slate-500 font-medium text-xs mt-1">Create Authorized Industrial Inspection Account</p>
        </div>

        {error && (
          <div className="mb-5 p-3.5 bg-red-50 text-red-700 rounded-xl text-xs font-medium border border-red-200 flex items-start gap-2.5 animate-fadeIn">
            <AlertCircle size={18} className="text-red-600 shrink-0 mt-0.5" />
            <span className="flex-1 leading-relaxed">{error}</span>
          </div>
        )}

        {isWarmingUp && (
          <div className="mb-5 p-3.5 bg-amber-50 text-amber-800 rounded-xl text-xs font-medium border border-amber-200 flex items-center gap-2.5">
            <Loader2 size={16} className="text-amber-600 animate-spin shrink-0" />
            <span>Connecting to cloud inspection database... please wait a moment.</span>
          </div>
        )}
        
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Username
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <User size={16} />
              </div>
              <input 
                {...register("username", { 
                  required: "Username is required",
                  minLength: { value: 3, message: "Username must be at least 3 characters" }
                })}
                type="text" 
                autoComplete="username"
                disabled={isLoading}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm transition-all text-slate-900 placeholder:text-slate-400 disabled:bg-slate-50"
                placeholder="quality_eng_01"
              />
            </div>
            {errors.username && (
              <p className="text-red-600 text-xs mt-1 font-medium">{errors.username.message}</p>
            )}
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Email Address
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <Mail size={16} />
              </div>
              <input 
                {...register("email", { 
                  required: "Email address is required",
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: "Invalid email address format"
                  }
                })}
                type="email" 
                autoComplete="email"
                disabled={isLoading}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm transition-all text-slate-900 placeholder:text-slate-400 disabled:bg-slate-50"
                placeholder="engineer@visioninspect.ai"
              />
            </div>
            {errors.email && (
              <p className="text-red-600 text-xs mt-1 font-medium">{errors.email.message}</p>
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
                  required: "Password is required",
                  minLength: { value: 6, message: "Password must be at least 6 characters" }
                })}
                type="password" 
                autoComplete="new-password"
                disabled={isLoading}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm transition-all text-slate-900 placeholder:text-slate-400 disabled:bg-slate-50"
                placeholder="••••••••"
              />
            </div>
            {errors.password && (
              <p className="text-red-600 text-xs mt-1 font-medium">{errors.password.message}</p>
            )}
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
              Platform Role
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                <Briefcase size={16} />
              </div>
              <select 
                {...register("role_name", { required: true })}
                disabled={isLoading}
                className="w-full pl-10 pr-3.5 py-2.5 rounded-xl border border-slate-300 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-sm transition-all text-slate-900 bg-white cursor-pointer disabled:bg-slate-50"
              >
                <option value="QUALITY_ENGINEER">Quality Engineer</option>
                <option value="SUPERVISOR">Supervisor</option>
                <option value="OPERATOR">Operator</option>
                <option value="ADMIN">Admin</option>
              </select>
            </div>
          </div>

          <button 
            type="submit" 
            disabled={isLoading}
            className="w-full bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white font-bold py-3 px-4 rounded-xl transition-all disabled:opacity-70 disabled:cursor-not-allowed flex justify-center items-center h-12 text-sm shadow-md hover:shadow-lg cursor-pointer mt-3 group"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <Loader2 size={18} className="animate-spin" />
                <span>Creating Database Account...</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span>Register Account</span>
                <ArrowRight size={16} className="group-hover:translate-x-0.5 transition-transform" />
              </div>
            )}
          </button>
        </form>
        
        <div className="mt-6 pt-5 border-t border-slate-100 text-center">
          <p className="text-xs text-slate-600 mb-2 font-medium">Already have an account?</p>
          <Link 
            href="/login"
            className="inline-flex items-center gap-1.5 text-sm text-blue-600 font-bold hover:text-blue-700 hover:underline cursor-pointer"
          >
            <span>Sign in to existing account</span>
            <ArrowRight size={14} />
          </Link>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-100 text-center">
          <p className="text-[11px] text-slate-400">VisionInspect AI • Industrial Access Control</p>
        </div>
      </div>
    </div>
  );
}
