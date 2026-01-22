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

@Component({
  selector: 'app-feedback-results',
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
    <div class="feedback-results-container">
      <div class="feedback-results-content">
        <div class="page-header">
          <h1 class="page-title">Результаты обратной связи</h1>
          <p class="page-subtitle">Анализ отзывов студентов</p>
        </div>

        <mat-card class="filter-card">
          <mat-card-content>
            <div class="filters">
              <mat-form-field appearance="outline">
                <mat-label>Курс</mat-label>
                <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="loadStats()">
                  <mat-option [value]="null">Все курсы</mat-option>
                  <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                    {{ subject.name }}
                  </mat-option>
                </mat-select>
              </mat-form-field>

              <mat-form-field appearance="outline" *ngIf="selectedSubjectId">
                <mat-label>Группа</mat-label>
                <mat-select [(ngModel)]="selectedGroupId" (selectionChange)="loadStats()">
                  <mat-option [value]="null">Все группы</mat-option>
                  <mat-option *ngFor="let group of groups" [value]="group.id">
                    {{ group.name }}
                  </mat-option>
                </mat-select>
              </mat-form-field>
            </div>
          </mat-card-content>
        </mat-card>

        <div *ngIf="loading" class="loading">
          <mat-spinner></mat-spinner>
        </div>

        <div *ngIf="!loading && stats">
          <mat-card class="stats-card">
            <mat-card-header>
              <mat-card-title>
                <mat-icon>bar_chart</mat-icon>
                Статистика отзывов
              </mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <div class="stats-grid">
                <div class="stat-item">
                  <div class="stat-value">{{ stats.total_responses }}</div>
                  <div class="stat-label">Всего отзывов</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ stats.avg_quality_rating.toFixed(1) }}</div>
                  <div class="stat-label">Качество обучения</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ stats.avg_content_rating.toFixed(1) }}</div>
                  <div class="stat-label">Содержание курса</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ stats.avg_materials_rating.toFixed(1) }}</div>
                  <div class="stat-label">Материалы</div>
                </div>
                <div class="stat-item">
                  <div class="stat-value">{{ stats.avg_support_rating.toFixed(1) }}</div>
                  <div class="stat-label">Поддержка</div>
                </div>
                <div class="stat-item highlight">
                  <div class="stat-value">{{ stats.overall_avg.toFixed(1) }}</div>
                  <div class="stat-label">Общая оценка</div>
                </div>
              </div>
            </mat-card-content>
          </mat-card>

          <mat-card class="feedbacks-card">
            <mat-card-header>
              <mat-card-title>
                <mat-icon>list</mat-icon>
                Все отзывы
              </mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <div *ngIf="feedbacks.length === 0" class="empty-state">
                Нет отзывов
              </div>
              <table mat-table [dataSource]="feedbacks" *ngIf="feedbacks.length > 0" class="feedbacks-table">
                <ng-container matColumnDef="user_name">
                  <th mat-header-cell *matHeaderCellDef>Студент</th>
                  <td mat-cell *matCellDef="let feedback">{{ feedback.user_name }}</td>
                </ng-container>
                <ng-container matColumnDef="quality_rating">
                  <th mat-header-cell *matHeaderCellDef>Качество</th>
                  <td mat-cell *matCellDef="let feedback">{{ feedback.quality_rating }}/5</td>
                </ng-container>
                <ng-container matColumnDef="content_rating">
                  <th mat-header-cell *matHeaderCellDef>Содержание</th>
                  <td mat-cell *matCellDef="let feedback">{{ feedback.content_rating }}/5</td>
                </ng-container>
                <ng-container matColumnDef="materials_rating">
                  <th mat-header-cell *matHeaderCellDef>Материалы</th>
                  <td mat-cell *matCellDef="let feedback">{{ feedback.materials_rating }}/5</td>
                </ng-container>
                <ng-container matColumnDef="support_rating">
                  <th mat-header-cell *matHeaderCellDef>Поддержка</th>
                  <td mat-cell *matCellDef="let feedback">{{ feedback.support_rating }}/5</td>
                </ng-container>
                <ng-container matColumnDef="comment">
                  <th mat-header-cell *matHeaderCellDef>Комментарий</th>
                  <td mat-cell *matCellDef="let feedback">{{ feedback.comment || '—' }}</td>
                </ng-container>
                <ng-container matColumnDef="created_at">
                  <th mat-header-cell *matHeaderCellDef>Дата</th>
                  <td mat-cell *matCellDef="let feedback">{{ formatDate(feedback.created_at) }}</td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="feedbackColumns"></tr>
                <tr mat-row *matRowDef="let row; columns: feedbackColumns;"></tr>
              </table>
            </mat-card-content>
          </mat-card>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .feedback-results-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .feedback-results-content {
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

    .stats-card, .feedbacks-card {
      margin-bottom: 24px;
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      background: white;
    }

    .stats-card mat-card-title {
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

    .stat-item.highlight {
      background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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

    .feedbacks-card mat-card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 20px;
      font-weight: 500;
    }

    .feedbacks-table {
      width: 100%;
    }

    .empty-state {
      text-align: center;
      padding: 40px;
      color: #999;
    }

    @media (max-width: 768px) {
      .feedback-results-container {
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
export class FeedbackResultsComponent implements OnInit {
  subjects: any[] = [];
  groups: any[] = [];
  selectedSubjectId: string | null = null;
  selectedGroupId: string | null = null;
  stats: any = null;
  feedbacks: any[] = [];
  loading = false;
  feedbackColumns = ['user_name', 'quality_rating', 'content_rating', 'materials_rating', 'support_rating', 'comment', 'created_at'];

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.loadSubjects();
    this.loadStats();
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

  loadStats() {
    this.loading = true;
    if (this.selectedSubjectId) {
      this.loadGroups();
    } else {
      this.groups = [];
      this.selectedGroupId = null;
    }

    this.apiService.getFeedbackStats(
      this.selectedSubjectId || undefined,
      this.selectedGroupId || undefined
    ).subscribe({
      next: (stats) => {
        this.stats = stats;
        this.loadFeedbacks();
      },
      error: (err) => {
        console.error('Error loading feedback stats:', err);
        this.loading = false;
      }
    });
  }

  loadFeedbacks() {
    this.apiService.getFeedbacks(
      undefined,
      this.selectedSubjectId || undefined,
      this.selectedGroupId || undefined
    ).subscribe({
      next: (feedbacks) => {
        this.feedbacks = feedbacks;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading feedbacks:', err);
        this.loading = false;
      }
    });
  }

  formatDate(dateString: string): string {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch {
      return '';
    }
  }
}

