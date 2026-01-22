import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { RussianDatePipe } from '../../core/pipes/russian-date.pipe';

@Component({
  selector: 'app-tests',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTooltipModule,
    RussianDatePipe
  ],
  template: `
    <div class="tests-container">
      <div class="tests-content">
        <div class="page-header">
          <h1 class="page-title">Тесты</h1>
          <p class="page-subtitle">Создавайте и проходите тесты</p>
        </div>
      
      <button mat-raised-button color="primary" (click)="createTest()" class="create-button">
        <mat-icon>add</mat-icon>
        Создать тест
      </button>

      <div *ngIf="tests.length === 0" class="empty-state">
        <mat-icon>quiz</mat-icon>
        <p>Нет созданных тестов</p>
        <p class="empty-hint">Создайте первый тест, чтобы начать</p>
      </div>

      <div class="tests-grid" *ngIf="tests.length > 0">
        <mat-card *ngFor="let test of tests" class="test-card">
          <mat-card-header>
            <mat-card-title>
              <mat-icon class="test-icon">assignment</mat-icon>
              {{ test.title }}
            </mat-card-title>
            <mat-card-subtitle class="test-subtitle">
              <mat-icon class="subtitle-icon">category</mat-icon>
              {{ getTestTypeLabel(test.test_type) }}
            </mat-card-subtitle>
          </mat-card-header>
          <mat-card-content>
            <p class="test-description">{{ test.description || 'Нет описания' }}</p>
            <div class="test-info">
              <div class="info-item" *ngIf="test.due_date">
                <mat-icon>schedule</mat-icon>
                <span>Дедлайн: {{ test.due_date | russianDate:'datetime' }}</span>
              </div>
              <div class="info-item">
                <mat-icon>help_outline</mat-icon>
                <span>Вопросов: {{ test.questions?.length || 0 }}</span>
              </div>
            </div>
          </mat-card-content>
          <mat-card-actions class="test-actions">
            <button mat-button [routerLink]="['/tests', test.id]" class="action-btn">
              <mat-icon>visibility</mat-icon>
              Просмотр
            </button>
            <button 
              mat-raised-button 
              color="primary" 
              [routerLink]="['/tests', test.id, 'take']" 
              [disabled]="isTestExpired(test)"
              class="action-btn"
              [matTooltip]="isTestExpired(test) ? 'Тест недоступен: срок прохождения истек' : ''">
              <mat-icon>play_arrow</mat-icon>
              {{ isCompleted(test.id) ? 'Пройти снова' : 'Пройти' }}
            </button>
            <span *ngIf="isTestExpired(test)" class="badge-expired">
              <mat-icon>error</mat-icon> Недоступен
            </span>
            <span *ngIf="isCompleted(test.id)" class="badge-done">
              <mat-icon>check_circle</mat-icon> Пройден
            </span>
            <button mat-icon-button color="warn" (click)="deleteTest(test.id)" class="delete-btn">
              <mat-icon>delete</mat-icon>
            </button>
          </mat-card-actions>
        </mat-card>
      </div>
      </div>
    </div>
  `,
  styles: [`
    .tests-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .tests-content {
      max-width: 1200px;
      margin: 0 auto;
    }

    .page-header {
      margin-bottom: 32px;
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

    .create-button {
      margin-bottom: 32px;
      padding: 12px 32px;
      font-size: 16px;
      font-weight: 500;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }

    .tests-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 24px;
    }

    .test-card {
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: all 0.3s ease;
      display: flex;
      flex-direction: column;
      background: white;
      overflow: hidden;
    }

    .test-card:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
      transform: translateY(-4px);
    }

    .test-icon {
      margin-right: 8px;
      color: #3f51b5;
      vertical-align: middle;
    }

    .test-subtitle {
      display: flex;
      align-items: center;
      margin-top: 8px;
    }

    .subtitle-icon {
      font-size: 16px;
      width: 16px;
      height: 16px;
      margin-right: 4px;
      color: #999;
    }

    .test-description {
      color: #666;
      margin-bottom: 16px;
      line-height: 1.5;
    }

    .test-info {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .info-item {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #666;
      font-size: 14px;
    }

    .info-item mat-icon {
      font-size: 18px;
      width: 18px;
      height: 18px;
      color: #999;
    }

    .test-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 16px 16px 16px;
      margin-top: auto;
    }

    .action-btn {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .delete-btn {
      margin-left: auto;
    }
    .badge-done {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: #4caf50;
      font-weight: 600;
      margin-left: 8px;
    }
    .badge-expired {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      color: #f44336;
      font-weight: 600;
      margin-left: 8px;
    }

    .empty-state {
      text-align: center;
      padding: 80px 20px;
      color: #999;
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
      font-size: 16px !important;
      color: #9e9e9e;
      margin: 0;
    }

    @media (max-width: 768px) {
      .tests-container {
        padding: 16px;
      }

      .tests-grid {
        grid-template-columns: 1fr;
      }

      .page-title {
        font-size: 24px;
      }
    }
  `]
})
export class TestsComponent implements OnInit {
  tests: any[] = [];
  completedByUser: Set<string> = new Set();

  constructor(
    private apiService: ApiService,
    private router: Router,
    private auth: AuthService
  ) {}

  ngOnInit() {
    this.loadTests();
  }

  loadTests() {
    this.apiService.getTests().subscribe({
      next: (tests) => {
        this.tests = tests;
        this.loadUserSubmissions();
      },
      error: (err) => console.error('Error loading tests:', err)
    });
  }

  loadUserSubmissions() {
    const user = this.auth.getCurrentUser();
    if (!user) return;
    this.apiService.getSubmissions(undefined, user.name).subscribe({
      next: (subs) => {
        this.completedByUser = new Set(subs.map((s) => s.test_id));
      },
      error: (err) => console.error('Error loading user submissions', err)
    });
  }

  createTest() {
    this.router.navigate(['/tests/create']);
  }

  deleteTest(id: string) {
    if (confirm('Удалить тест?')) {
      this.apiService.deleteTest(id).subscribe({
        next: () => this.loadTests(),
        error: (err) => {
          console.error('Error deleting test:', err);
          alert('Ошибка при удалении теста');
        }
      });
    }
  }

  getTestTypeLabel(type: string): string {
    return type === 'multiple_choice' ? 'С вариантами ответов' : 'С ключевыми словами';
  }

  isCompleted(testId: string): boolean {
    return this.completedByUser.has(testId);
  }

  isTestExpired(test: any): boolean {
    if (!test.due_date) {
      return false; // Если дедлайн не установлен, тест доступен
    }
    try {
      // Дата хранится в московском времени (формат +03:00)
      // Просто сравниваем напрямую
      const dueDate = new Date(test.due_date);
      const now = new Date();
      
      // Тест недоступен только если дедлайн уже прошел
      return dueDate.getTime() < now.getTime();
    } catch (e) {
      console.error('Error checking test expiration:', e, test);
      return false; // В случае ошибки считаем тест доступным
    }
  }
}

