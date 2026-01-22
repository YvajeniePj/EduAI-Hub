import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatFormFieldModule } from '@angular/material/form-field';
import { ApiService } from '../../core/services/api.service';
import { RussianDatePipe } from '../../core/pipes/russian-date.pipe';

@Component({
  selector: 'app-admin-db',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatInputModule,
    MatTabsModule,
    MatTableModule,
    MatIconModule,
    MatTooltipModule,
    MatFormFieldModule,
    RussianDatePipe
  ],
  template: `
    <div class="admin-container">
      <div class="admin-header">
        <h1 class="admin-title">Управление базой данных</h1>
        <p class="admin-subtitle">Просмотр и управление всеми сущностями системы</p>
      </div>

      <mat-tab-group class="admin-tabs">
        <!-- Users Tab -->
        <mat-tab label="Пользователи">
          <div class="tab-content">
            <div class="section-header">
              <h2>Пользователи</h2>
              <div class="inline-form">
                <mat-form-field appearance="outline">
                  <mat-label>Имя пользователя</mat-label>
                  <input matInput [(ngModel)]="newUserName" />
                </mat-form-field>
                <button mat-raised-button color="primary" (click)="createUser()" [disabled]="!newUserName.trim()">
                  <mat-icon>add</mat-icon>
                  Создать
                </button>
              </div>
            </div>
            <mat-card class="data-card">
              <table mat-table [dataSource]="users" class="admin-table">
                <ng-container matColumnDef="id">
                  <th mat-header-cell *matHeaderCellDef>ID</th>
                  <td mat-cell *matCellDef="let u" class="id-cell">{{ truncateId(u.id) }}</td>
                </ng-container>
                <ng-container matColumnDef="name">
                  <th mat-header-cell *matHeaderCellDef>Имя</th>
                  <td mat-cell *matCellDef="let u" class="name-cell">{{ u.name }}</td>
                </ng-container>
                <ng-container matColumnDef="actions">
                  <th mat-header-cell *matHeaderCellDef>Действия</th>
                  <td mat-cell *matCellDef="let u" class="actions-cell">
                    <button mat-icon-button color="warn" (click)="deleteUser(u.id)" matTooltip="Удалить пользователя">
                      <mat-icon>delete</mat-icon>
                    </button>
                  </td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="['id', 'name', 'actions']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['id', 'name', 'actions'];"></tr>
                <tr class="mat-row" *matNoDataRow>
                  <td class="mat-cell" [attr.colspan]="3">Нет пользователей</td>
                </tr>
              </table>
            </mat-card>
          </div>
        </mat-tab>

        <!-- Subjects Tab -->
        <mat-tab label="Курсы">
          <div class="tab-content">
            <div class="section-header">
              <h2>Курсы</h2>
              <p class="count-badge">Всего: {{ subjects.length }}</p>
            </div>
            <mat-card class="data-card">
              <table mat-table [dataSource]="subjects" class="admin-table">
                <ng-container matColumnDef="id">
                  <th mat-header-cell *matHeaderCellDef>ID</th>
                  <td mat-cell *matCellDef="let s" class="id-cell">{{ truncateId(s.id) }}</td>
                </ng-container>
                <ng-container matColumnDef="name">
                  <th mat-header-cell *matHeaderCellDef>Название</th>
                  <td mat-cell *matCellDef="let s" class="name-cell">{{ s.name }}</td>
                </ng-container>
                <ng-container matColumnDef="description">
                  <th mat-header-cell *matHeaderCellDef>Описание</th>
                  <td mat-cell *matCellDef="let s" class="description-cell">{{ s.description || '—' }}</td>
                </ng-container>
                <ng-container matColumnDef="created_at">
                  <th mat-header-cell *matHeaderCellDef>Создан</th>
                  <td mat-cell *matCellDef="let s" class="date-cell">{{ s.created_at | russianDate:'datetime' }}</td>
                </ng-container>
                <ng-container matColumnDef="actions">
                  <th mat-header-cell *matHeaderCellDef>Действия</th>
                  <td mat-cell *matCellDef="let s" class="actions-cell">
                    <button mat-icon-button color="warn" (click)="deleteSubject(s.id)" matTooltip="Удалить курс">
                      <mat-icon>delete</mat-icon>
                    </button>
                  </td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="['id', 'name', 'description', 'created_at', 'actions']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['id', 'name', 'description', 'created_at', 'actions'];"></tr>
              </table>
            </mat-card>
          </div>
        </mat-tab>

        <!-- Tests Tab -->
        <mat-tab label="Тесты">
          <div class="tab-content">
            <div class="section-header">
              <h2>Тесты</h2>
              <p class="count-badge">Всего: {{ tests.length }}</p>
            </div>
            <mat-card class="data-card">
              <table mat-table [dataSource]="tests" class="admin-table">
                <ng-container matColumnDef="id">
                  <th mat-header-cell *matHeaderCellDef>ID</th>
                  <td mat-cell *matCellDef="let t" class="id-cell">{{ truncateId(t.id) }}</td>
                </ng-container>
                <ng-container matColumnDef="title">
                  <th mat-header-cell *matHeaderCellDef>Название</th>
                  <td mat-cell *matCellDef="let t" class="name-cell">{{ t.title }}</td>
                </ng-container>
                <ng-container matColumnDef="test_type">
                  <th mat-header-cell *matHeaderCellDef>Тип</th>
                  <td mat-cell *matCellDef="let t" class="type-cell">
                    <span [class]="'type-badge type-' + t.test_type">
                      {{ t.test_type === 'multiple_choice' ? 'С вариантами' : 'С ключевыми словами' }}
                    </span>
                  </td>
                </ng-container>
                <ng-container matColumnDef="questions_count">
                  <th mat-header-cell *matHeaderCellDef>Вопросов</th>
                  <td mat-cell *matCellDef="let t" class="number-cell">{{ t.questions?.length || 0 }}</td>
                </ng-container>
                <ng-container matColumnDef="due_date">
                  <th mat-header-cell *matHeaderCellDef>Дедлайн</th>
                  <td mat-cell *matCellDef="let t" class="date-cell">
                    {{ t.due_date ? (t.due_date | russianDate:'datetime') : '—' }}
                  </td>
                </ng-container>
                <ng-container matColumnDef="time_limit">
                  <th mat-header-cell *matHeaderCellDef>Ограничение времени</th>
                  <td mat-cell *matCellDef="let t" class="number-cell">
                    {{ t.time_limit_minutes ? t.time_limit_minutes + ' мин' : '—' }}
                  </td>
                </ng-container>
                <ng-container matColumnDef="created_at">
                  <th mat-header-cell *matHeaderCellDef>Создан</th>
                  <td mat-cell *matCellDef="let t" class="date-cell">{{ t.created_at | russianDate:'datetime' }}</td>
                </ng-container>
                <ng-container matColumnDef="actions">
                  <th mat-header-cell *matHeaderCellDef>Действия</th>
                  <td mat-cell *matCellDef="let t" class="actions-cell">
                    <button mat-icon-button color="warn" (click)="deleteTest(t.id)" matTooltip="Удалить тест">
                      <mat-icon>delete</mat-icon>
                    </button>
                  </td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="['id', 'title', 'test_type', 'questions_count', 'due_date', 'time_limit', 'created_at', 'actions']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['id', 'title', 'test_type', 'questions_count', 'due_date', 'time_limit', 'created_at', 'actions'];"></tr>
              </table>
            </mat-card>
          </div>
        </mat-tab>

        <!-- Materials Tab -->
        <mat-tab label="Материалы">
          <div class="tab-content">
            <div class="section-header">
              <h2>Материалы</h2>
              <p class="count-badge">Всего: {{ materials.length }}</p>
            </div>
            <mat-card class="data-card">
              <table mat-table [dataSource]="materials" class="admin-table">
                <ng-container matColumnDef="id">
                  <th mat-header-cell *matHeaderCellDef>ID</th>
                  <td mat-cell *matCellDef="let m" class="id-cell">{{ truncateId(m.id) }}</td>
                </ng-container>
                <ng-container matColumnDef="name">
                  <th mat-header-cell *matHeaderCellDef>Имя файла</th>
                  <td mat-cell *matCellDef="let m" class="name-cell">{{ m.original_name || m.name }}</td>
                </ng-container>
                <ng-container matColumnDef="size">
                  <th mat-header-cell *matHeaderCellDef>Размер</th>
                  <td mat-cell *matCellDef="let m" class="number-cell">{{ formatSize(m.size) }}</td>
                </ng-container>
                <ng-container matColumnDef="mime_type">
                  <th mat-header-cell *matHeaderCellDef>Тип</th>
                  <td mat-cell *matCellDef="let m" class="type-cell">{{ m.mime_type }}</td>
                </ng-container>
                <ng-container matColumnDef="uploader">
                  <th mat-header-cell *matHeaderCellDef>Загрузил</th>
                  <td mat-cell *matCellDef="let m" class="name-cell">{{ m.uploader }}</td>
                </ng-container>
                <ng-container matColumnDef="note">
                  <th mat-header-cell *matHeaderCellDef>Примечание</th>
                  <td mat-cell *matCellDef="let m" class="description-cell">{{ m.note || '—' }}</td>
                </ng-container>
                <ng-container matColumnDef="created_at">
                  <th mat-header-cell *matHeaderCellDef>Создан</th>
                  <td mat-cell *matCellDef="let m" class="date-cell">{{ m.created_at | russianDate:'datetime' }}</td>
                </ng-container>
                <ng-container matColumnDef="actions">
                  <th mat-header-cell *matHeaderCellDef>Действия</th>
                  <td mat-cell *matCellDef="let m" class="actions-cell">
                    <button mat-icon-button color="warn" (click)="deleteMaterial(m.id)" matTooltip="Удалить материал">
                      <mat-icon>delete</mat-icon>
                    </button>
                  </td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="['id', 'name', 'size', 'mime_type', 'uploader', 'note', 'created_at', 'actions']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['id', 'name', 'size', 'mime_type', 'uploader', 'note', 'created_at', 'actions'];"></tr>
              </table>
            </mat-card>
          </div>
        </mat-tab>

        <!-- Submissions Tab -->
        <mat-tab label="Сдачи тестов">
          <div class="tab-content">
            <div class="section-header">
              <h2>Сдачи тестов</h2>
              <p class="count-badge">Всего: {{ submissions.length }}</p>
            </div>
            <mat-card class="data-card">
              <table mat-table [dataSource]="submissions" class="admin-table">
                <ng-container matColumnDef="id">
                  <th mat-header-cell *matHeaderCellDef>ID</th>
                  <td mat-cell *matCellDef="let s" class="id-cell">{{ truncateId(s.id) }}</td>
                </ng-container>
                <ng-container matColumnDef="user">
                  <th mat-header-cell *matHeaderCellDef>Пользователь</th>
                  <td mat-cell *matCellDef="let s" class="name-cell">{{ s.user }}</td>
                </ng-container>
                <ng-container matColumnDef="test_id">
                  <th mat-header-cell *matHeaderCellDef>ID Теста</th>
                  <td mat-cell *matCellDef="let s" class="id-cell">{{ truncateId(s.test_id) }}</td>
                </ng-container>
                <ng-container matColumnDef="score">
                  <th mat-header-cell *matHeaderCellDef>Оценка</th>
                  <td mat-cell *matCellDef="let s" class="score-cell">
                    <span [class.score-full]="s.total_score === s.total_max && s.total_max > 0"
                          [class.score-partial]="s.total_score > 0 && s.total_score < s.total_max"
                          [class.score-zero]="s.total_score === 0">
                      {{ s.total_score }} / {{ s.total_max }}
                    </span>
                  </td>
                </ng-container>
                <ng-container matColumnDef="points_awarded">
                  <th mat-header-cell *matHeaderCellDef>Очки</th>
                  <td mat-cell *matCellDef="let s" class="number-cell">{{ s.points_awarded }}</td>
                </ng-container>
                <ng-container matColumnDef="is_finished">
                  <th mat-header-cell *matHeaderCellDef>Статус</th>
                  <td mat-cell *matCellDef="let s" class="status-cell">
                    <span [class.status-finished]="s.is_finished === 'true'" 
                          [class.status-pending]="s.is_finished === 'false'">
                      {{ s.is_finished === 'true' ? 'Завершена' : 'В процессе' }}
                    </span>
                  </td>
                </ng-container>
                <ng-container matColumnDef="started_at">
                  <th mat-header-cell *matHeaderCellDef>Начато</th>
                  <td mat-cell *matCellDef="let s" class="date-cell">{{ s.started_at | russianDate:'datetime' }}</td>
                </ng-container>
                <ng-container matColumnDef="finished_at">
                  <th mat-header-cell *matHeaderCellDef>Завершено</th>
                  <td mat-cell *matCellDef="let s" class="date-cell">
                    {{ s.finished_at ? (s.finished_at | russianDate:'datetime') : '—' }}
                  </td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="['id', 'user', 'test_id', 'score', 'points_awarded', 'is_finished', 'started_at', 'finished_at']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['id', 'user', 'test_id', 'score', 'points_awarded', 'is_finished', 'started_at', 'finished_at'];"></tr>
              </table>
            </mat-card>
          </div>
        </mat-tab>

        <!-- Reviews Tab -->
        <mat-tab label="Отзывы">
          <div class="tab-content">
            <div class="section-header">
              <h2>Отзывы (кросс-проверка)</h2>
              <p class="count-badge">Всего: {{ reviews.length }}</p>
            </div>
            <mat-card class="data-card">
              <table mat-table [dataSource]="reviews" class="admin-table">
                <ng-container matColumnDef="id">
                  <th mat-header-cell *matHeaderCellDef>ID</th>
                  <td mat-cell *matCellDef="let r" class="id-cell">{{ truncateId(r.id) }}</td>
                </ng-container>
                <ng-container matColumnDef="submission_id">
                  <th mat-header-cell *matHeaderCellDef>ID Сдачи</th>
                  <td mat-cell *matCellDef="let r" class="id-cell">{{ truncateId(r.submission_id) }}</td>
                </ng-container>
                <ng-container matColumnDef="reviewer">
                  <th mat-header-cell *matHeaderCellDef>Рецензент</th>
                  <td mat-cell *matCellDef="let r" class="name-cell">{{ r.reviewer }}</td>
                </ng-container>
                <ng-container matColumnDef="avg_score">
                  <th mat-header-cell *matHeaderCellDef>Средний балл</th>
                  <td mat-cell *matCellDef="let r" class="score-cell">{{ r.avg_score.toFixed(2) }}</td>
                </ng-container>
                <ng-container matColumnDef="scores">
                  <th mat-header-cell *matHeaderCellDef>Оценки</th>
                  <td mat-cell *matCellDef="let r" class="scores-cell">
                    Релевантность: {{ r.relevance }}, Структура: {{ r.structure }},<br>
                    Аргументация: {{ r.argument }}, Ясность: {{ r.clarity }}
                  </td>
                </ng-container>
                <ng-container matColumnDef="created_at">
                  <th mat-header-cell *matHeaderCellDef>Создан</th>
                  <td mat-cell *matCellDef="let r" class="date-cell">{{ r.created_at | russianDate:'datetime' }}</td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="['id', 'submission_id', 'reviewer', 'avg_score', 'scores', 'created_at']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['id', 'submission_id', 'reviewer', 'avg_score', 'scores', 'created_at'];"></tr>
              </table>
            </mat-card>
          </div>
        </mat-tab>

        <!-- Videos Tab -->
        <mat-tab label="Видео">
          <div class="tab-content">
            <div class="section-header">
              <h2>Видео материалы</h2>
              <p class="count-badge">Всего: {{ videos.length }}</p>
            </div>
            <mat-card class="data-card">
              <table mat-table [dataSource]="videos" class="admin-table">
                <ng-container matColumnDef="id">
                  <th mat-header-cell *matHeaderCellDef>ID</th>
                  <td mat-cell *matCellDef="let v" class="id-cell">{{ truncateId(v.id) }}</td>
                </ng-container>
                <ng-container matColumnDef="title">
                  <th mat-header-cell *matHeaderCellDef>Название</th>
                  <td mat-cell *matCellDef="let v" class="name-cell">{{ v.title }}</td>
                </ng-container>
                <ng-container matColumnDef="url">
                  <th mat-header-cell *matHeaderCellDef>URL</th>
                  <td mat-cell *matCellDef="let v" class="url-cell">
                    <a [href]="v.url" target="_blank" class="url-link">{{ truncateUrl(v.url) }}</a>
                  </td>
                </ng-container>
                <ng-container matColumnDef="uploader">
                  <th mat-header-cell *matHeaderCellDef>Загрузил</th>
                  <td mat-cell *matCellDef="let v" class="name-cell">{{ v.uploader }}</td>
                </ng-container>
                <ng-container matColumnDef="created_at">
                  <th mat-header-cell *matHeaderCellDef>Создан</th>
                  <td mat-cell *matCellDef="let v" class="date-cell">{{ v.created_at | russianDate:'datetime' }}</td>
                </ng-container>
                <ng-container matColumnDef="actions">
                  <th mat-header-cell *matHeaderCellDef>Действия</th>
                  <td mat-cell *matCellDef="let v" class="actions-cell">
                    <button mat-icon-button color="warn" (click)="deleteVideo(v.id)" matTooltip="Удалить видео">
                      <mat-icon>delete</mat-icon>
                    </button>
                  </td>
                </ng-container>
                <tr mat-header-row *matHeaderRowDef="['id', 'title', 'url', 'uploader', 'created_at', 'actions']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['id', 'title', 'url', 'uploader', 'created_at', 'actions'];"></tr>
              </table>
            </mat-card>
          </div>
        </mat-tab>
      </mat-tab-group>
    </div>
  `,
  styles: [`
    .admin-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .admin-header {
      max-width: 1400px;
      margin: 0 auto 32px auto;
    }

    .admin-title {
      font-size: 32px;
      font-weight: 600;
      color: #1a237e;
      margin: 0 0 8px 0;
      line-height: 1.2;
    }

    .admin-subtitle {
      font-size: 16px;
      color: #616161;
      margin: 0;
      line-height: 1.5;
    }

    .admin-tabs {
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      overflow: hidden;
    }

    .tab-content {
      padding: 24px;
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }

    .section-header h2 {
      font-size: 24px;
      font-weight: 600;
      color: #1a237e;
      margin: 0;
    }

    .count-badge {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 6px 16px;
      border-radius: 16px;
      font-size: 14px;
      font-weight: 500;
      margin: 0;
    }

    .inline-form {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .data-card {
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      border-radius: 12px;
      overflow: hidden;
    }

    .admin-table {
      width: 100%;
    }

    .admin-table th {
      background: #f8f9fa;
      font-weight: 600;
      color: #424242;
      font-size: 14px;
      padding: 16px;
    }

    .admin-table td {
      padding: 12px 16px;
      font-size: 14px;
      border-bottom: 1px solid #e0e0e0;
    }

    .admin-table tr:hover {
      background: #f8f9fa;
    }

    .id-cell {
      font-family: 'Courier New', monospace;
      font-size: 12px;
      color: #666;
      max-width: 120px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .name-cell {
      font-weight: 500;
      color: #212121;
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .description-cell {
      color: #616161;
      max-width: 300px;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .date-cell {
      color: #616161;
      font-size: 13px;
      white-space: nowrap;
    }

    .number-cell {
      text-align: center;
      color: #424242;
      font-weight: 500;
    }

    .type-cell {
      text-align: center;
    }

    .type-badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;
    }

    .type-multiple_choice {
      background: #e3f2fd;
      color: #1976d2;
    }

    .type-keyword_based {
      background: #f3e5f5;
      color: #7b1fa2;
    }

    .score-cell {
      text-align: center;
      font-weight: 600;
    }

    .score-full {
      color: #4caf50;
    }

    .score-partial {
      color: #ff9800;
    }

    .score-zero {
      color: #f44336;
    }

    .status-cell {
      text-align: center;
    }

    .status-finished {
      color: #4caf50;
      font-weight: 500;
    }

    .status-pending {
      color: #ff9800;
      font-weight: 500;
    }

    .scores-cell {
      font-size: 12px;
      color: #616161;
      line-height: 1.6;
    }

    .url-cell {
      max-width: 300px;
    }

    .url-link {
      color: #667eea;
      text-decoration: none;
    }

    .url-link:hover {
      text-decoration: underline;
    }

    .actions-cell {
      text-align: center;
    }

    @media (max-width: 1200px) {
      .admin-container {
        padding: 16px;
      }

      .admin-title {
        font-size: 24px;
      }

      .admin-table {
        font-size: 12px;
      }

      .admin-table th,
      .admin-table td {
        padding: 8px 12px;
      }
    }
  `]
})
export class AdminDbComponent implements OnInit {
  users: any[] = [];
  subjects: any[] = [];
  materials: any[] = [];
  videos: any[] = [];
  tests: any[] = [];
  submissions: any[] = [];
  reviews: any[] = [];

  newUserName = '';

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadAll();
  }

  loadAll() {
    this.api.getUsers().subscribe((u) => (this.users = u));
    this.api.getSubjects().subscribe((s) => (this.subjects = s));
    this.api.getMaterials().subscribe((m) => (this.materials = m));
    this.api.getVideos().subscribe((v) => (this.videos = v));
    this.api.getTests().subscribe((t) => (this.tests = t));
    this.api.getSubmissions().subscribe((s) => (this.submissions = s));
    this.api.getReviews().subscribe((r) => (this.reviews = r));
  }

  createUser() {
    const name = this.newUserName.trim();
    if (!name) return;
    this.api.createUser(name).subscribe({
      next: (user) => {
        this.users = [...this.users, user];
        this.newUserName = '';
      },
      error: (err) => alert(err.error?.detail || 'Не удалось создать пользователя')
    });
  }

  deleteUser(id: string) {
    if (confirm('Удалить пользователя? Это действие нельзя отменить.')) {
      // Note: API doesn't have delete user endpoint, you may need to add it
      alert('Функция удаления пользователя не реализована в API');
    }
  }

  deleteSubject(id: string) {
    if (confirm('Удалить курс? Все связанные данные также будут удалены.')) {
      this.api.deleteSubject(id).subscribe({
        next: () => {
          this.subjects = this.subjects.filter(s => s.id !== id);
        },
        error: (err) => alert(err.error?.detail || 'Ошибка при удалении курса')
      });
    }
  }

  deleteTest(id: string) {
    if (confirm('Удалить тест? Все связанные данные также будут удалены.')) {
      this.api.deleteTest(id).subscribe({
        next: () => {
          this.tests = this.tests.filter(t => t.id !== id);
        },
        error: (err) => alert(err.error?.detail || 'Ошибка при удалении теста')
      });
    }
  }

  deleteMaterial(id: string) {
    if (confirm('Удалить материал?')) {
      this.api.deleteMaterial(id).subscribe({
        next: () => {
          this.materials = this.materials.filter(m => m.id !== id);
        },
        error: (err) => alert(err.error?.detail || 'Ошибка при удалении материала')
      });
    }
  }

  deleteVideo(id: string) {
    if (confirm('Удалить видео?')) {
      this.api.deleteVideo(id).subscribe({
        next: () => {
          this.videos = this.videos.filter(v => v.id !== id);
        },
        error: (err) => alert(err.error?.detail || 'Ошибка при удалении видео')
      });
    }
  }

  truncateId(id: string): string {
    if (!id) return '';
    const str = id.toString();
    return str.substring(0, 8) + '...';
  }

  truncateUrl(url: string): string {
    if (!url) return '';
    if (url.length > 50) {
      return url.substring(0, 50) + '...';
    }
    return url;
  }

  formatSize(bytes: number): string {
    if (!bytes) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }
}
