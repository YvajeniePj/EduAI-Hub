import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatTableModule } from '@angular/material/table';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatSelectModule,
    MatFormFieldModule,
    MatIconModule,
    MatTableModule,
    MatProgressSpinnerModule
  ],
  template: `
    <div class="analytics-container">
      <div class="analytics-content">
        <div class="page-header">
          <h1 class="page-title">Аналитика успеваемости</h1>
          <p class="page-subtitle">Отчеты по студентам и группам</p>
        </div>

        <mat-card class="filter-card">
          <mat-card-content>
            <div class="filters">
              <mat-form-field appearance="outline">
                <mat-label>Курс</mat-label>
                <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="loadReport()">
                  <mat-option [value]="null">Все курсы</mat-option>
                  <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                    {{ subject.name }}
                  </mat-option>
                </mat-select>
              </mat-form-field>

              <mat-form-field appearance="outline" *ngIf="selectedSubjectId">
                <mat-label>Группа</mat-label>
                <mat-select [(ngModel)]="selectedGroupId" (selectionChange)="loadReport()">
                  <mat-option [value]="null">Все группы</mat-option>
                  <mat-option *ngFor="let group of groups" [value]="group.id">
                    {{ group.name }}
                  </mat-option>
                </mat-select>
              </mat-form-field>

              <mat-form-field appearance="outline">
                <mat-label>Период (дней)</mat-label>
                <mat-select [(ngModel)]="selectedDays" (selectionChange)="loadReport()">
                  <mat-option [value]="7">7 дней</mat-option>
                  <mat-option [value]="30">30 дней</mat-option>
                  <mat-option [value]="90">90 дней</mat-option>
                  <mat-option [value]="365">Год</mat-option>
                </mat-select>
              </mat-form-field>

              <button mat-raised-button color="primary" (click)="loadReport()" [disabled]="loading">
                <mat-icon>refresh</mat-icon>
                Загрузить отчет
              </button>
            </div>
          </mat-card-content>
        </mat-card>

        <div *ngIf="loading" class="loading">
          <mat-spinner></mat-spinner>
        </div>

        <div *ngIf="!loading && !report && !errorMessage" class="empty-state">
          <mat-icon>assessment</mat-icon>
          <p>Нет данных для отображения</p>
          <p class="empty-hint">Выберите фильтры и загрузите отчет</p>
        </div>

        <div *ngIf="!loading && errorMessage" class="error-state">
          <mat-icon>error_outline</mat-icon>
          <p>{{ errorMessage }}</p>
        </div>

        <div *ngIf="!loading && report">
          <mat-card class="summary-card">
            <mat-card-header>
              <mat-card-title>
                <mat-icon>assessment</mat-icon>
                Общая статистика
              </mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <div class="stats-grid">
                <div class="stat-item">
                  <div class="stat-value">{{ report.total_students }}</div>
                  <div class="stat-label">Студентов</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ report.average_score.toFixed(1) }}</div>
                  <div class="stat-label">Средний балл</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ report.total_tests_completed }}</div>
                  <div class="stat-label">Пройдено тестов</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ report.total_time_hours.toFixed(1) }} ч</div>
                  <div class="stat-label">Время в системе</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ report.total_logins }}</div>
                  <div class="stat-label">Входов в систему</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ report.engagement_score.toFixed(1) }}%</div>
                  <div class="stat-label">Вовлеченность</div>
                </div>
              </div>
            </mat-card-content>
          </mat-card>

          <mat-card class="progress-card">
            <mat-card-header>
              <mat-card-title>
                <mat-icon>trending_up</mat-icon>
                Прогресс студентов
              </mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <div *ngIf="progressList.length === 0" class="empty-state">
                Нет данных о прогрессе
              </div>
              <table mat-table [dataSource]="progressList" *ngIf="progressList.length > 0" class="progress-table">
                <ng-container matColumnDef="user_name">
                  <th mat-header-cell *matHeaderCellDef>Студент</th>
                  <td mat-cell *matCellDef="let progress">{{ progress.user_name }}</td>
                </ng-container>
                <ng-container matColumnDef="tests_completed">
                  <th mat-header-cell *matHeaderCellDef>Тестов пройдено</th>
                  <td mat-cell *matCellDef="let progress">{{ progress.tests_completed }} / {{ progress.tests_total }}</td>
                </ng-container>
                <ng-container matColumnDef="average_score">
                  <th mat-header-cell *matHeaderCellDef>Средний балл</th>
                  <td mat-cell *matCellDef="let progress">
                    {{ progress.average_score ? progress.average_score.toFixed(1) : '—' }}
                  </td>
                </ng-container>
                <ng-container matColumnDef="total_time_hours">
                  <th mat-header-cell *matHeaderCellDef>Время в системе</th>
                  <td mat-cell *matCellDef="let progress">
                    {{ (progress.total_time_seconds / 3600).toFixed(1) }} ч
                  </td>
                </ng-container>
                <ng-container matColumnDef="materials_viewed">
                  <th mat-header-cell *matHeaderCellDef>Материалов просмотрено</th>
                  <td mat-cell *matCellDef="let progress">{{ progress.materials_viewed }}</td>
                </ng-container>
                <ng-container matColumnDef="videos_viewed">
                  <th mat-header-cell *matHeaderCellDef>Видео просмотрено</th>
                  <td mat-cell *matCellDef="let progress">{{ progress.videos_viewed }}</td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="progressColumns"></tr>
                <tr mat-row *matRowDef="let row; columns: progressColumns;"></tr>
              </table>
            </mat-card-content>
          </mat-card>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .analytics-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .analytics-content {
      max-width: 1400px;
      margin: 0 auto;
    }

    .page-header {
      margin-bottom: 32px;
    }

    .page-title {
      font-size: 32px;
      font-weight: 600;
      margin: 0 0 8px 0;
      color: #1a237e;
      line-height: 1.2;
    }

    .page-subtitle {
      font-size: 16px;
      color: #616161;
      margin: 0;
      line-height: 1.5;
    }

    .filter-card {
      margin-bottom: 24px;
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      background: white;
    }

    .filters {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }

    .loading {
      display: flex;
      justify-content: center;
      padding: 60px;
    }

    .summary-card, .progress-card {
      margin-bottom: 24px;
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      background: white;
    }

    .summary-card mat-card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 20px;
      font-weight: 500;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 24px;
      margin-top: 16px;
    }

    .stat-item {
      text-align: center;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 12px;
      color: white;
    }

    .stat-value {
      font-size: 32px;
      font-weight: 600;
      margin-bottom: 8px;
    }

    .stat-label {
      font-size: 14px;
      opacity: 0.9;
    }

    .progress-card mat-card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 20px;
      font-weight: 500;
    }

    .progress-table {
      width: 100%;
    }

    .empty-state, .error-state {
      text-align: center;
      padding: 80px 20px;
      color: #999;
    }

    .empty-state mat-icon, .error-state mat-icon {
      font-size: 96px;
      width: 96px;
      height: 96px;
      margin-bottom: 24px;
      opacity: 0.4;
      color: #9e9e9e;
    }

    .error-state mat-icon {
      color: #f44336;
    }

    .empty-state p, .error-state p {
      font-size: 24px;
      font-weight: 500;
      color: #616161;
      margin: 0 0 8px 0;
    }

    .empty-hint {
      font-size: 16px !important;
      color: #9e9e9e;
      margin: 0;
    }

    @media (max-width: 768px) {
      .analytics-container {
        padding: 16px;
      }

      .filters {
        grid-template-columns: 1fr;
      }

      .stats-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }
  `]
})
export class AnalyticsComponent implements OnInit {
  subjects: any[] = [];
  groups: any[] = [];
  selectedSubjectId: string | null = null;
  selectedGroupId: string | null = null;
  selectedDays: number = 30;
  report: any = null;
  progressList: any[] = [];
  loading = false;
  errorMessage: string | null = null;
  hasLoadedOnce = false;
  progressColumns = ['user_name', 'tests_completed', 'average_score', 'total_time_hours', 'materials_viewed', 'videos_viewed'];

  constructor(
    private apiService: ApiService,
    private authService: AuthService
  ) {}

  ngOnInit() {
    this.loadSubjects();
    // Не загружаем отчет сразу, пусть пользователь выберет фильтры
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  loadGroups() {
    if (!this.selectedSubjectId) {
      this.groups = [];
      return;
    }
    this.apiService.getGroups(this.selectedSubjectId).subscribe({
      next: (groups) => {
        this.groups = groups;
      },
      error: (err) => {
        console.error('Error loading groups:', err);
        this.groups = [];
      }
    });
  }

  loadReport() {
    this.loading = true;
    this.errorMessage = null;
    this.hasLoadedOnce = true;
    if (this.selectedSubjectId) {
      this.loadGroups();
    } else {
      this.groups = [];
      this.selectedGroupId = null;
    }

    this.apiService.getAnalyticsReport(
      this.selectedSubjectId || undefined,
      this.selectedGroupId || undefined,
      undefined,
      this.selectedDays
    ).subscribe({
      next: (report) => {
        console.log('Analytics report loaded:', report);
        this.report = report;
        this.loadProgress();
      },
      error: (err) => {
        console.error('Error loading analytics report:', err);
        this.errorMessage = err.error?.detail || err.message || 'Ошибка при загрузке отчета';
        this.loading = false;
        this.report = null;
        this.progressList = [];
      }
    });
  }

  loadProgress() {
    this.apiService.getProgress(
      undefined,
      this.selectedSubjectId || undefined,
      this.selectedGroupId || undefined
    ).subscribe({
      next: (progress) => {
        this.progressList = progress;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading progress:', err);
        this.loading = false;
      }
    });
  }
}

