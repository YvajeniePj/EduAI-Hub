import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatTabsModule } from '@angular/material/tabs';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-peer-review',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatButtonModule,
    MatInputModule,
    MatSelectModule,
    MatListModule,
    MatIconModule,
    MatTabsModule
  ],
  template: `
    <div class="peer-review-container">
      <h1>Кросс-проверка (анонимно)</h1>
      
      <!-- Шаг 1: Выбор курса -->
      <mat-card *ngIf="step === 1">
        <mat-card-content>
          <h2>Шаг 1: Выберите курс</h2>
          <mat-form-field style="width: 100%;">
            <mat-label>Выберите курс</mat-label>
            <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="onSubjectSelected()">
              <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                {{ subject.name }}
              </mat-option>
            </mat-select>
          </mat-form-field>
        </mat-card-content>
      </mat-card>

      <!-- Шаг 2: Выбор теста -->
      <mat-card *ngIf="step === 2">
        <mat-card-content>
          <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <button mat-icon-button (click)="goToStep(1)" style="margin-right: 10px;">
              <mat-icon>arrow_back</mat-icon>
            </button>
            <h2 style="margin: 0;">Шаг 2: Выберите тест</h2>
          </div>
          <mat-form-field style="width: 100%;">
            <mat-label>Выберите тест</mat-label>
            <mat-select [(ngModel)]="selectedTestId" (selectionChange)="onTestSelected()">
              <mat-option *ngFor="let test of tests" [value]="test.id">
                {{ test.title }}
              </mat-option>
            </mat-select>
          </mat-form-field>
        </mat-card-content>
      </mat-card>

      <!-- Шаг 3: Список пользователей с баллами -->
      <mat-card *ngIf="step === 3">
        <mat-card-content>
          <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <button mat-icon-button (click)="goToStep(2)" style="margin-right: 10px;">
              <mat-icon>arrow_back</mat-icon>
            </button>
            <h2 style="margin: 0;">Шаг 3: Выберите работу для проверки</h2>
          </div>
          <div *ngIf="submissionsForReview.length === 0" class="info-message">
            Нет доступных работ для проверки
          </div>
          <mat-list *ngIf="submissionsForReview.length > 0">
            <mat-list-item *ngFor="let submission of submissionsForReview" 
                          (click)="onSubmissionSelected(submission)"
                          style="cursor: pointer; border: 1px solid #e0e0e0; border-radius: 4px; margin-bottom: 10px; padding: 10px;">
              <div style="width: 100%;">
                <div style="font-weight: 500; margin-bottom: 5px;">
                  Пользователь: {{ submission.user }}
                </div>
                <div style="color: #666; font-size: 14px;">
                  Баллы: {{ submission.total_score }} / {{ submission.total_max }}
                  <span *ngIf="submission.total_max > 0" style="margin-left: 10px;">
                    ({{ (submission.total_score / submission.total_max * 100).toFixed(1) }}%)
                  </span>
                </div>
                <div *ngIf="submission.assignment" style="color: #666; font-size: 14px; margin-top: 5px;">
                  Задание: {{ submission.assignment }}
                </div>
              </div>
            </mat-list-item>
          </mat-list>
        </mat-card-content>
      </mat-card>

      <!-- Шаг 4: Форма оценки -->
      <div *ngIf="step === 4">
        <mat-card>
          <mat-card-content>
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
              <button mat-icon-button (click)="goToStep(3)" style="margin-right: 10px;">
                <mat-icon>arrow_back</mat-icon>
              </button>
              <h2 style="margin: 0;">Шаг 4: Оценка работы</h2>
            </div>
            
            <div *ngIf="selectedSubmission">
              <div style="margin-bottom: 20px;">
                <p><strong>Пользователь:</strong> {{ selectedSubmission.user }}</p>
                <p><strong>Баллы:</strong> {{ selectedSubmission.total_score }} / {{ selectedSubmission.total_max }}</p>
                <p *ngIf="selectedSubmission.assignment"><strong>Задание:</strong> {{ selectedSubmission.assignment }}</p>
              </div>

              <div *ngIf="selectedSubmission.answers && selectedSubmission.answers.length > 0" style="margin-bottom: 20px;">
                <h3>Ответы:</h3>
                <div *ngFor="let answer of selectedSubmission.answers" style="margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 4px;">
                  <div style="font-weight: 500; margin-bottom: 5px;">Вопрос: {{ answer.question_id }}</div>
                  <div>{{ answer.answer || '—' }}</div>
                </div>
              </div>
              <div *ngIf="!selectedSubmission.answers || selectedSubmission.answers.length === 0" style="margin-bottom: 20px;">
                <p>Ответ отсутствует</p>
              </div>

              <!-- Форма с обычными input полями -->
              <form [formGroup]="reviewForm!" *ngIf="reviewForm">
                <h3>Оцените по критериям (1–5)</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                  <mat-form-field>
                    <mat-label>Соответствие заданию</mat-label>
                    <input matInput 
                           type="number" 
                           formControlName="relevance" 
                           min="1" 
                           max="5" 
                           step="1"
                           (blur)="validateScore('relevance')"
                           required>
                    <mat-hint>Оценка от 1 до 5</mat-hint>
                    <mat-error *ngIf="reviewForm.get('relevance')?.hasError('min') || reviewForm.get('relevance')?.hasError('max')">
                      Оценка должна быть от 1 до 5
                    </mat-error>
                  </mat-form-field>
                  
                  <mat-form-field>
                    <mat-label>Структура и логика</mat-label>
                    <input matInput 
                           type="number" 
                           formControlName="structure" 
                           min="1" 
                           max="5" 
                           step="1"
                           (blur)="validateScore('structure')"
                           required>
                    <mat-hint>Оценка от 1 до 5</mat-hint>
                    <mat-error *ngIf="reviewForm.get('structure')?.hasError('min') || reviewForm.get('structure')?.hasError('max')">
                      Оценка должна быть от 1 до 5
                    </mat-error>
                  </mat-form-field>
                  
                  <mat-form-field>
                    <mat-label>Аргументация / примеры</mat-label>
                    <input matInput 
                           type="number" 
                           formControlName="argument" 
                           min="1" 
                           max="5" 
                           step="1"
                           (blur)="validateScore('argument')"
                           required>
                    <mat-hint>Оценка от 1 до 5</mat-hint>
                    <mat-error *ngIf="reviewForm.get('argument')?.hasError('min') || reviewForm.get('argument')?.hasError('max')">
                      Оценка должна быть от 1 до 5
                    </mat-error>
                  </mat-form-field>
                  
                  <mat-form-field>
                    <mat-label>Ясность изложения</mat-label>
                    <input matInput 
                           type="number" 
                           formControlName="clarity" 
                           min="1" 
                           max="5" 
                           step="1"
                           (blur)="validateScore('clarity')"
                           required>
                    <mat-hint>Оценка от 1 до 5</mat-hint>
                    <mat-error *ngIf="reviewForm.get('clarity')?.hasError('min') || reviewForm.get('clarity')?.hasError('max')">
                      Оценка должна быть от 1 до 5
                    </mat-error>
                  </mat-form-field>
                </div>

                <mat-form-field style="width: 100%; margin-top: 20px;">
                  <mat-label>Комментарий (опционально)</mat-label>
                  <textarea matInput formControlName="comment" rows="3"></textarea>
                </mat-form-field>

                <div style="margin-top: 10px; margin-bottom: 20px;">
                  <strong>Средняя оценка: {{ getAverageScore() }} / 5</strong>
                </div>

                <button mat-raised-button color="primary" (click)="submitReview()" [disabled]="!reviewForm.valid">
                  Отправить отзыв
                </button>
              </form>
            </div>
          </mat-card-content>
        </mat-card>

      </div>
    </div>
  `,
  styles: [`
    .peer-review-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    mat-card {
      margin-bottom: 20px;
    }
    .info-message {
      padding: 20px;
      text-align: center;
      color: #666;
    }
    mat-list-item:hover {
      background-color: #f5f5f5;
    }
  `]
})
export class PeerReviewComponent implements OnInit {
  subjects: any[] = [];
  tests: any[] = [];
  submissionsForReview: any[] = [];
  selectedSubjectId: string = '';
  selectedTestId: string = '';
  selectedSubmission: any = null;
  reviewForm: FormGroup | null = null;
  currentUser: string = '';
  step: number = 1; // 1 - выбор курса, 2 - выбор теста, 3 - выбор пользователя, 4 - форма оценки

  constructor(
    private apiService: ApiService, 
    private auth: AuthService,
    private fb: FormBuilder,
    private cdr: ChangeDetectorRef
  ) {
    // Не создаем форму в конструкторе, создадим позже
  }

  ngOnInit() {
    const user = this.auth.getCurrentUser();
    this.currentUser = user?.name || '';
    if (!this.currentUser) {
      alert('Войдите, чтобы проверять работы');
      return;
    }
    this.loadSubjects();
  }

  goToStep(stepNumber: number) {
    this.step = stepNumber;
    if (stepNumber === 1) {
      this.selectedSubjectId = '';
      this.selectedTestId = '';
      this.selectedSubmission = null;
      this.tests = [];
      this.submissionsForReview = [];
    } else if (stepNumber === 2) {
      this.selectedTestId = '';
      this.selectedSubmission = null;
      this.submissionsForReview = [];
    } else if (stepNumber === 3) {
      this.selectedSubmission = null;
      this.reviewForm = null;
    }
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  onSubjectSelected() {
    if (!this.selectedSubjectId) return;
    this.loadTests();
  }

  loadTests() {
    if (!this.selectedSubjectId) return;
    
    this.apiService.getTests(this.selectedSubjectId).subscribe({
      next: (tests) => {
        this.tests = tests;
        if (tests.length > 0) {
          this.step = 2;
        }
      },
      error: (err) => console.error('Error loading tests:', err)
    });
  }

  onTestSelected() {
    if (!this.selectedTestId) return;
    this.loadSubmissionsForReview();
  }

  loadSubmissionsForReview() {
    if (!this.selectedTestId) return;
    if (!this.currentUser) {
      alert('Войдите, чтобы проверять работы');
      return;
    }

    this.apiService.getSubmissionsForReview(this.selectedTestId, this.currentUser).subscribe({
      next: (submissions) => {
        // Фильтруем только завершенные работы
        this.submissionsForReview = submissions.filter(s => s.is_finished === 'true');
        this.step = 3;
      },
      error: (err) => {
        console.error('Error loading submissions:', err);
        this.submissionsForReview = [];
      }
    });
  }

  onSubmissionSelected(submission: any) {
    this.selectedSubmission = submission;
    this.step = 4;
    
    // Создаем форму сразу с начальными значениями
    this.initializeForm();
  }

  initializeForm() {
    // Создаем форму с начальными значениями 1
    this.reviewForm = this.fb.group({
      relevance: [1, [Validators.required, Validators.min(1), Validators.max(5)]],
      structure: [1, [Validators.required, Validators.min(1), Validators.max(5)]],
      argument: [1, [Validators.required, Validators.min(1), Validators.max(5)]],
      clarity: [1, [Validators.required, Validators.min(1), Validators.max(5)]],
      comment: ['']
    });
    
    // Подписываемся на изменения значений для автоматического обновления отображения
    this.reviewForm.valueChanges.subscribe(() => {
      this.cdr.detectChanges();
    });
  }

  validateScore(controlName: string) {
    // Валидация и корректировка значения при потере фокуса
    if (this.reviewForm) {
      const control = this.reviewForm.get(controlName);
      if (control) {
        let value = control.value;
        if (value === null || value === undefined || value === '') {
          value = 1;
        } else {
          value = Number(value);
          if (isNaN(value) || value < 1) {
            value = 1;
          } else if (value > 5) {
            value = 5;
          }
        }
        control.setValue(value, { emitEvent: true });
      }
    }
  }

  getAverageScore(): number {
    if (!this.reviewForm) return 0;
    
    const relevance = this.reviewForm.get('relevance')?.value || 1;
    const structure = this.reviewForm.get('structure')?.value || 1;
    const argument = this.reviewForm.get('argument')?.value || 1;
    const clarity = this.reviewForm.get('clarity')?.value || 1;
    
    return Math.round(((relevance + structure + argument + clarity) / 4) * 100) / 100;
  }

  submitReview() {
    if (!this.currentUser) {
      alert('Войдите, чтобы отправить отзыв');
      return;
    }
    
    if (!this.reviewForm || !this.reviewForm.valid) {
      alert('Пожалуйста, заполните все поля оценки');
      return;
    }

    if (!this.selectedSubmission) {
      alert('Работа не выбрана');
      return;
    }

    const review = {
      submission_id: this.selectedSubmission.id,
      assignment_id: this.selectedTestId,
      reviewer: this.currentUser,
      relevance: this.reviewForm.get('relevance')?.value,
      structure: this.reviewForm.get('structure')?.value,
      argument: this.reviewForm.get('argument')?.value,
      clarity: this.reviewForm.get('clarity')?.value,
      comment: this.reviewForm.get('comment')?.value || ''
    };

    this.apiService.createReview(review).subscribe({
      next: () => {
        alert('Отзыв сохранен! Вам начислено +1 очко за кросс-проверку.');
        // Возвращаемся к списку пользователей
        this.goToStep(3);
        // Перезагружаем список, чтобы исключить проверенную работу (если нужно)
        this.loadSubmissionsForReview();
      },
      error: (err) => {
        console.error('Error creating review:', err);
        alert('Ошибка при сохранении отзыва: ' + (err.error?.detail || err.message));
      }
    });
  }
}
