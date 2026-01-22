import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, MatCardModule, MatButtonModule, MatInputModule, MatFormFieldModule, MatSelectModule, RouterModule],
  template: `
    <div class="auth-container">
      <mat-card>
        <mat-card-header>
          <mat-card-title>Регистрация</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <mat-form-field appearance="outline" style="width: 100%;">
            <mat-label>Логин (имя)</mat-label>
            <input matInput [(ngModel)]="name" (keyup.enter)="register()" />
          </mat-form-field>
          <mat-form-field appearance="outline" style="width: 100%;">
            <mat-label>Роль</mat-label>
            <mat-select [(ngModel)]="role">
              <mat-option value="student">Студент</mat-option>
              <mat-option value="teacher">Преподаватель</mat-option>
            </mat-select>
          </mat-form-field>
          <div class="actions">
            <button mat-raised-button color="primary" (click)="register()" [disabled]="!name.trim() || loading">
              {{ loading ? 'Создание...' : 'Зарегистрироваться' }}
            </button>
            <button mat-button color="accent" routerLink="/login">Уже есть аккаунт</button>
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
export class RegisterComponent {
  name = '';
  role = 'student';
  loading = false;

  constructor(private auth: AuthService, private router: Router) { }

  register() {
    const trimmed = this.name.trim();
    if (!trimmed) return;
    this.loading = true;
    this.auth.register(trimmed, this.role).subscribe({
      next: () => {
        this.loading = false;
        this.router.navigate(['/']);
      },
      error: (err) => {
        this.loading = false;
        let errorMessage = 'Ошибка при регистрации.';

        if (err.status === 409) {
          errorMessage = 'Имя уже занято. Выберите другое имя или войдите.';
        } else if (err.status === 503) {
          errorMessage = 'Сервис временно недоступен. Попробуйте позже.';
        } else if (err.error?.detail) {
          errorMessage = err.error.detail;
        }

        alert(errorMessage);
        console.error('Register error:', err);
      }
    });
  }
}

