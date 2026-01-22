import { Injectable } from '@angular/core';
import { ApiService } from './api.service';
import { tap, catchError, map } from 'rxjs/operators';
import { Observable, of, BehaviorSubject, throwError } from 'rxjs';
import { HttpClient, HttpHeaders, HttpBackend } from '@angular/common/http';

export interface CurrentUser {
  id: string;
  name: string;
  avatar_url?: string;
  role?: string;
}

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private storageKey = 'eduai-jwt-token';
  private userStorageKey = 'eduai-current-user';
  private currentUserSubject = new BehaviorSubject<CurrentUser | null>(this.getCurrentUserFromStorage());
  private tokenSubject = new BehaviorSubject<string | null>(this.getTokenFromStorage());

  private apiBaseUrl = '/api';
  private httpWithoutInterceptor: HttpClient;

  // Observable для подписки на изменения текущего пользователя
  currentUser$ = this.currentUserSubject.asObservable();
  token$ = this.tokenSubject.asObservable();

  constructor(
    private http: HttpClient,
    private handler: HttpBackend
  ) {
    this.httpWithoutInterceptor = new HttpClient(handler);

    // Инициализируем текущего пользователя при создании сервиса
    const user = this.getCurrentUserFromStorage();
    const token = this.getTokenFromStorage();

    if (user && token) {
      this.currentUserSubject.next(user);
      this.tokenSubject.next(token);
      // Проверяем валидность токена
      this.validateToken();
    }
  }

  // ...

  logout() {
    const currentUser = this.getCurrentUser();
    if (currentUser) {
      // Track logout activity
      try {
        this.httpWithoutInterceptor.post(`${this.apiBaseUrl}/analytics/activities`, {
          user_name: currentUser.name,
          action_type: 'logout',
          resource_type: 'system',
          session_duration: null
        }).subscribe({
          error: (err) => console.error('Error tracking logout activity:', err)
        });
      } catch (err) {
        console.error('Error tracking logout activity:', err);
      }
    }
    localStorage.removeItem(this.storageKey);
    localStorage.removeItem(this.userStorageKey);
    this.currentUserSubject.next(null);
    this.tokenSubject.next(null);
  }

  // ...

  loginByName(name: string): Observable<CurrentUser> {
    return this.http.post<{ token: string, user: CurrentUser }>(`${this.apiBaseUrl}/auth/login`, { name }).pipe(
      tap((response) => {
        this.setToken(response.token);
        this.setCurrentUser(response.user);
        // Track login activity
        this.httpWithoutInterceptor.post(`${this.apiBaseUrl}/analytics/activities`, {
          user_name: response.user.name,
          action_type: 'login',
          resource_type: 'system',
          session_duration: null
        }).subscribe({
          error: (err) => console.error('Error tracking login activity:', err)
        });
      }),
      map((response) => response.user)
    );
  }

  private getTokenFromStorage(): string | null {
    return localStorage.getItem(this.storageKey);
  }

  private getCurrentUserFromStorage(): CurrentUser | null {
    const raw = localStorage.getItem(this.userStorageKey);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }

  getToken(): string | null {
    return this.tokenSubject.value;
  }

  getCurrentUser(): CurrentUser | null {
    return this.currentUserSubject.value;
  }

  private setToken(token: string) {
    localStorage.setItem(this.storageKey, token);
    this.tokenSubject.next(token);
  }

  private setCurrentUser(user: CurrentUser) {
    localStorage.setItem(this.userStorageKey, JSON.stringify(user));
    this.currentUserSubject.next(user);
  }

  private validateToken() {
    const token = this.getToken();
    if (!token) {
      this.logout();
      return;
    }

    // Проверяем токен через API
    this.http.get<any>('/api/auth/me', {
      headers: new HttpHeaders({
        'Authorization': `Bearer ${token}`
      })
    }).pipe(
      catchError((error) => {
        // Токен невалиден только при 401 или 403
        if (error.status === 401 || error.status === 403) {
          this.logout();
          return throwError(() => new Error('Invalid token'));
        }
        // При ошибках сети или 5xx нe выходим
        console.warn('Token validation failed temporarily:', error);
        return of(null); // Return observable to keep stream alive (though current logic ends here)
      })
    ).subscribe({
      next: (userData) => {
        if (userData) {
          // Токен валиден, обновляем пользователя если нужно
          const currentUser = this.getCurrentUser();
          if (!currentUser || currentUser.id !== userData.user_id) {
            this.setCurrentUser({
              id: userData.user_id,
              name: userData.username,
              avatar_url: userData.avatar_url,
              role: userData.role
            });
          }
        }
      },
      error: () => {
        // This shouldn't be reached if we handle catchError correctly, 
        // but just in case of unhandled error
        console.error('Unhandled token validation error');
      }
    });
  }



  register(name: string, role: string = 'student'): Observable<CurrentUser> {
    return this.http.post<{ token: string, user: CurrentUser }>('/api/auth/register', { name, role }).pipe(
      tap((response) => {
        this.setToken(response.token);
        this.setCurrentUser(response.user);
      }),
      map((response) => response.user)
    );
  }

  isAuthenticated(): boolean {
    return this.getToken() !== null && this.getCurrentUser() !== null;
  }
}

