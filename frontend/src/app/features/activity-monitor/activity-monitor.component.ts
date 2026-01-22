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
  selector: 'app-activity-monitor',
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
    <div class="monitor-container">
      <div class="monitor-content">
        <div class="page-header">
          <h1 class="page-title">Мониторинг активности</h1>
          <p class="page-subtitle">Отслеживание активности студентов в системе</p>
        </div>

        <mat-card class="filter-card">
          <mat-card-content>
            <div class="filters">
              <mat-form-field appearance="outline">
                <mat-label>Студент</mat-label>
                <mat-select [(ngModel)]="selectedUserName" (selectionChange)="loadActivityStats()">
                  <mat-option [value]="null">Все студенты</mat-option>
                  <mat-option *ngFor="let user of users" [value]="user.name">
                    {{ user.name }}
                  </mat-option>
                </mat-select>
              </mat-form-field>

              <mat-form-field appearance="outline">
                <mat-label>Период (дней)</mat-label>
                <mat-select [(ngModel)]="selectedDays" (selectionChange)="loadActivityStats()">
                  <mat-option [value]="7">7 дней</mat-option>
                  <mat-option [value]="30">30 дней</mat-option>
                  <mat-option [value]="90">90 дней</mat-option>
                </mat-select>
              </mat-form-field>
            </div>
          </mat-card-content>
        </mat-card>

        <div *ngIf="loading" class="loading">
          <mat-spinner></mat-spinner>
        </div>

        <div *ngIf="!loading">
          <mat-card class="stats-card" *ngIf="selectedUserName && activityStats.length > 0">
            <mat-card-header>
              <mat-card-title>
                <mat-icon>timeline</mat-icon>
                Статистика активности: {{ selectedUserName }}
              </mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <div class="activity-summary">
                <div class="summary-item">
                  <mat-icon>access_time</mat-icon>
                  <div>
                    <div class="summary-value">{{ getTotalTime() }} ч</div>
                    <div class="summary-label">Общее время</div>
                  </div>
                </div>
                <div class="summary-item">
                  <mat-icon>login</mat-icon>
                  <div>
                    <div class="summary-value">{{ getTotalLogins() }}</div>
                    <div class="summary-label">Входов</div>
                  </div>
                </div>
                <div class="summary-item">
                  <mat-icon>quiz</mat-icon>
                  <div>
                    <div class="summary-value">{{ getTotalTestActions() }}</div>
                    <div class="summary-label">Действий с тестами</div>
                  </div>
                </div>
                <div class="summary-item">
                  <mat-icon>description</mat-icon>
                  <div>
                    <div class="summary-value">{{ getTotalMaterialViews() }}</div>
                    <div class="summary-label">Просмотров материалов</div>
                  </div>
                </div>
                <div class="summary-item">
                  <mat-icon>play_circle</mat-icon>
                  <div>
                    <div class="summary-value">{{ getTotalVideoViews() }}</div>
                    <div class="summary-label">Просмотров видео</div>
                  </div>
                </div>
              </div>
            </mat-card-content>
          </mat-card>

          <mat-card class="activity-card">
            <mat-card-header>
              <mat-card-title>
                <mat-icon>history</mat-icon>
                История активности
              </mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <div *ngIf="activityStats.length === 0" class="empty-state">
                Нет данных об активности
              </div>
              <table mat-table [dataSource]="activityStats" *ngIf="activityStats.length > 0" class="activity-table">
                <ng-container matColumnDef="date">
                  <th mat-header-cell *matHeaderCellDef>Дата</th>
                  <td mat-cell *matCellDef="let stat">{{ stat.date }}</td>
                </ng-container>
                <ng-container matColumnDef="total_time">
                  <th mat-header-cell *matHeaderCellDef>Время в системе</th>
                  <td mat-cell *matCellDef="let stat">{{ formatTime(stat.total_time_seconds) }}</td>
                </ng-container>
                <ng-container matColumnDef="login_count">
                  <th mat-header-cell *matHeaderCellDef>Входов</th>
                  <td mat-cell *matCellDef="let stat">{{ stat.login_count }}</td>
                </ng-container>
                <ng-container matColumnDef="test_actions">
                  <th mat-header-cell *matHeaderCellDef>Действий с тестами</th>
                  <td mat-cell *matCellDef="let stat">{{ stat.test_actions }}</td>
                </ng-container>
                <ng-container matColumnDef="material_views">
                  <th mat-header-cell *matHeaderCellDef>Просмотров материалов</th>
                  <td mat-cell *matCellDef="let stat">{{ stat.material_views }}</td>
                </ng-container>
                <ng-container matColumnDef="video_views">
                  <th mat-header-cell *matHeaderCellDef>Просмотров видео</th>
                  <td mat-cell *matCellDef="let stat">{{ stat.video_views }}</td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="activityColumns"></tr>
                <tr mat-row *matRowDef="let row; columns: activityColumns;"></tr>
              </table>
            </mat-card-content>
          </mat-card>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .monitor-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .monitor-content {
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

    .stats-card, .activity-card {
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

    .activity-summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }

    .summary-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 12px;
      color: white;
    }

    .summary-item mat-icon {
      font-size: 32px;
      width: 32px;
      height: 32px;
    }

    .summary-value {
      font-size: 24px;
      font-weight: 600;
      margin-bottom: 4px;
    }

    .summary-label {
      font-size: 12px;
      opacity: 0.9;
    }

    .activity-card mat-card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 20px;
      font-weight: 500;
    }

    .activity-table {
      width: 100%;
    }

    .empty-state {
      text-align: center;
      padding: 40px;
      color: #999;
    }

    @media (max-width: 768px) {
      .monitor-container {
        padding: 16px;
      }

      .filters {
        grid-template-columns: 1fr;
      }

      .activity-summary {
        grid-template-columns: 1fr;
      }
    }
  `]
})
export class ActivityMonitorComponent implements OnInit {
  users: any[] = [];
  selectedUserName: string | null = null;
  selectedDays: number = 30;
  activityStats: any[] = [];
  loading = false;
  activityColumns = ['date', 'total_time', 'login_count', 'test_actions', 'material_views', 'video_views'];

  constructor(
    private apiService: ApiService,
    private authService: AuthService
  ) {}

  ngOnInit() {
    this.loadUsers();
    const currentUser = this.authService.getCurrentUser();
    if (currentUser) {
      this.selectedUserName = currentUser.name;
      this.loadActivityStats();
    }
  }

  loadUsers() {
    this.apiService.getUsers().subscribe({
      next: (users) => {
        this.users = users;
      },
      error: (err) => console.error('Error loading users:', err)
    });
  }

  loadActivityStats() {
    if (!this.selectedUserName) {
      this.activityStats = [];
      return;
    }

    this.loading = true;
    this.apiService.getActivityStats(this.selectedUserName, this.selectedDays).subscribe({
      next: (stats) => {
        this.activityStats = stats;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading activity stats:', err);
        this.loading = false;
      }
    });
  }

  formatTime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours} ч ${minutes} мин`;
    }
    return `${minutes} мин`;
  }

  getTotalTime(): number {
    return this.activityStats.reduce((sum, stat) => sum + stat.total_time_seconds, 0) / 3600;
  }

  getTotalLogins(): number {
    return this.activityStats.reduce((sum, stat) => sum + stat.login_count, 0);
  }

  getTotalTestActions(): number {
    return this.activityStats.reduce((sum, stat) => sum + stat.test_actions, 0);
  }

  getTotalMaterialViews(): number {
    return this.activityStats.reduce((sum, stat) => sum + stat.material_views, 0);
  }

  getTotalVideoViews(): number {
    return this.activityStats.reduce((sum, stat) => sum + stat.video_views, 0);
  }
}

