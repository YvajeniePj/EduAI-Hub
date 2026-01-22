import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule, MatCardModule, MatButtonModule, MatInputModule, MatFormFieldModule, RouterModule],
  template: `
    <div class="auth-container">
      <mat-card>
        <mat-card-header>
          <mat-card-title>Вход</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <mat-form-field appearance="outline" style="width: 100%;">
            <mat-label>Логин (имя)</mat-label>
            <input matInput [(ngModel)]="name" (keyup.enter)="login()" />
          </mat-form-field>
          <div class="actions">
            <button mat-raised-button color="primary" (click)="login()" [disabled]="!name.trim() || loading">
              {{ loading ? 'Входим...' : 'Войти' }}
            </button>
            <button mat-button color="accent" routerLink="/register">Регистрация</button>
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [`
    .auth-container {
      max-width: 420px;
      margin: 40px auto;
      padding: 0 16px;
    }
    .actions {
      display: flex;
      gap: 12px;
      margin-top: 12px;
    }
  `]
})
export class LoginComponent {
  name = '';
  loading = false;

  constructor(private auth: AuthService, private router: Router) { }

  login() {
    const trimmed = this.name.trim();
    if (!trimmed) return;
    this.loading = true;
    this.auth.loginByName(trimmed).subscribe({
      next: () => {
        this.loading = false;
        this.router.navigate(['/']);
      },
      error: (err) => {
        this.loading = false;
        let errorMessage = 'Ошибка при входе.';

        if (err.status === 404) {
          errorMessage = 'Пользователь не найден. Зарегистрируйтесь.';
        } else if (err.status === 503) {
          errorMessage = 'Сервис временно недоступен. Попробуйте позже.';
        } else if (err.error?.detail) {
          errorMessage = err.error.detail;
        }

        alert(errorMessage);
        console.error('Login error:', err);
      }
    });
  }
}

