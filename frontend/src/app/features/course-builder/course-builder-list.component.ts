import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-course-builder-list',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  template: `
    <div class="builder-list-container">
      <div class="page-header">
        <h1 class="page-title">Конструктор курсов</h1>
        <p class="page-subtitle">Создавайте и редактируйте структуру ваших курсов</p>
      </div>

      <div *ngIf="loading" class="loading">
        <mat-spinner></mat-spinner>
      </div>

      <div *ngIf="!loading && subjects.length === 0" class="empty-state">
        <mat-icon>school</mat-icon>
        <p>Нет созданных курсов</p>
        <p class="empty-hint">Создайте курс на странице "Курсы", чтобы начать работу с конструктором</p>
        <button mat-raised-button color="primary" routerLink="/subjects" style="margin-top: 16px;">
          <mat-icon>add</mat-icon>
          Создать курс
        </button>
      </div>

      <div class="subjects-grid" *ngIf="!loading && subjects.length > 0">
        <mat-card *ngFor="let subject of subjects" class="subject-card">
          <mat-card-header class="subject-header">
            <div class="subject-icon-wrapper">
              <mat-icon class="subject-icon">book</mat-icon>
            </div>
            <div class="subject-info">
              <mat-card-title class="subject-title">{{ subject.name }}</mat-card-title>
              <mat-card-subtitle *ngIf="subject.description" class="subject-description">
                {{ subject.description }}
              </mat-card-subtitle>
            </div>
          </mat-card-header>
          <mat-card-content *ngIf="subject.description" class="subject-content">
            <p class="description-text">{{ subject.description }}</p>
          </mat-card-content>
          <mat-card-actions class="subject-actions">
            <button mat-raised-button color="primary" [routerLink]="['/course-builder', subject.id]" class="builder-button">
              <mat-icon>edit</mat-icon>
              Открыть конструктор
            </button>
          </mat-card-actions>
        </mat-card>
      </div>
    </div>
  `,
  styles: [`
    .builder-list-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .page-header {
      max-width: 1200px;
      margin: 0 auto 32px auto;
    }

    .page-title {
      font-size: 32px;
      font-weight: 600;
      color: #1a237e;
      margin: 0 0 8px 0;
      line-height: 1.2;
    }

    .page-subtitle {
      font-size: 16px;
      color: #616161;
      margin: 0;
      line-height: 1.5;
    }

    .loading {
      display: flex;
      justify-content: center;
      padding: 60px;
    }

    .subjects-grid {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 24px;
    }

    .subject-card {
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: all 0.3s ease;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      background: white;
    }

    .subject-card:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
      transform: translateY(-4px);
    }

    .subject-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 24px;
      display: flex;
      align-items: flex-start;
      gap: 16px;
    }

    .subject-icon-wrapper {
      flex-shrink: 0;
    }

    .subject-icon {
      font-size: 40px;
      width: 40px;
      height: 40px;
      color: white;
    }

    .subject-info {
      flex: 1;
      min-width: 0;
    }

    .subject-title {
      font-size: 20px;
      font-weight: 600;
      margin: 0 0 8px 0;
      color: white;
      line-height: 1.3;
    }

    .subject-description {
      font-size: 14px;
      color: rgba(255, 255, 255, 0.9);
      margin: 0;
      line-height: 1.5;
      overflow: hidden;
      text-overflow: ellipsis;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }

    .subject-content {
      padding: 20px 24px;
      flex: 1;
    }

    .description-text {
      font-size: 15px;
      color: #424242;
      line-height: 1.6;
      margin: 0;
    }

    .subject-actions {
      padding: 16px 24px;
      border-top: 1px solid #e0e0e0;
      display: flex;
      justify-content: center;
    }

    .builder-button {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-size: 16px;
      padding: 12px;
    }

    .empty-state {
      max-width: 1200px;
      margin: 0 auto;
      text-align: center;
      padding: 80px 20px;
    }

    .empty-state mat-icon {
      font-size: 96px;
      width: 96px;
      height: 96px;
      margin-bottom: 24px;
      opacity: 0.4;
      color: #9e9e9e;
    }

    .empty-state p {
      font-size: 24px;
      font-weight: 500;
      color: #616161;
      margin: 0 0 8px 0;
    }

    .empty-hint {
      font-size: 16px;
      color: #9e9e9e;
      margin: 0 0 24px 0;
    }

    @media (max-width: 768px) {
      .builder-list-container {
        padding: 16px;
      }

      .subjects-grid {
        grid-template-columns: 1fr;
      }

      .page-title {
        font-size: 24px;
      }
    }
  `]
})
export class CourseBuilderListComponent implements OnInit {
  subjects: any[] = [];
  loading = false;

  constructor(
    private apiService: ApiService,
    private router: Router
  ) {}

  ngOnInit() {
    this.loadSubjects();
  }

  loadSubjects() {
    this.loading = true;
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading subjects:', err);
        this.loading = false;
      }
    });
  }
}
