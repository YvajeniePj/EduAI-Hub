import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { FormsModule, FormBuilder, FormGroup, FormArray, Validators, ReactiveFormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatRadioModule } from '@angular/material/radio';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatIconModule } from '@angular/material/icon';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { RussianDatePipe } from '../../core/pipes/russian-date.pipe';
import { interval, Subscription } from 'rxjs';

@Component({
  selector: 'app-test-take',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatRadioModule,
    MatCheckboxModule,
    MatProgressBarModule,
    MatIconModule,
    RussianDatePipe
  ],
  template: `
    <div class="test-container" *ngIf="test">
      <div *ngIf="isTestExpired" class="error-container">
        <div class="error-content">
          <mat-icon class="error-icon">error_outline</mat-icon>
          <h2>Тест недоступен</h2>
          <p>Срок прохождения этого теста истек. Дедлайн: {{ test.due_date | russianDate:'datetime' }}</p>
          <button mat-raised-button color="primary" routerLink="/tests" class="back-button">
            Вернуться к тестам
          </button>
        </div>
      </div>

      <div *ngIf="!isTestExpired" class="test-content">
        <div class="test-header">
          <div class="header-info">
            <h1 class="test-title">{{ test.title }}</h1>
            <p class="test-description" *ngIf="test.description">{{ test.description }}</p>
          </div>
          <div class="timer-container" *ngIf="hasTimeLimit">
            <div class="timer" [class.timer-warning]="timeRemaining <= 300" [class.timer-critical]="timeRemaining <= 60">
              <mat-icon class="timer-icon">schedule</mat-icon>
              <div class="timer-display">
                <span class="timer-value">{{ formatTime(timeRemaining) }}</span>
                <span class="timer-label">Осталось времени</span>
              </div>
            </div>
          </div>
        </div>

        <div class="test-main-layout">
          <form [formGroup]="answerForm" (ngSubmit)="onSubmit()" class="test-form">
            <div formArrayName="answers" class="questions-carousel">
              <div class="carousel-wrapper">
                <div class="carousel-track" [style.transform]="'translateX(-' + (currentQuestionIndex * 100) + '%)'">
                  <mat-card *ngFor="let question of test.questions; let i = index" 
                            class="question-card" 
                            [class.active]="i === currentQuestionIndex">
                    <mat-card-header class="question-header">
                      <div class="question-number">Вопрос {{ i + 1 }} из {{ test.questions.length }}</div>
                      <div class="question-points">{{ question.max_points }} баллов</div>
                    </mat-card-header>
                    <mat-card-content class="question-content">
                      <h3 class="question-title">{{ question.title }}</h3>

                      <!-- Multiple Choice -->
                      <div *ngIf="test.test_type === 'multiple_choice'" class="answer-section">
                        <mat-radio-group [formControlName]="i" 
                                          class="radio-group"
                                          (change)="onAnswerChange(i)">
                          <mat-radio-button *ngFor="let option of question.options; let j = index" 
                                            [value]="option" 
                                            class="radio-option">
                            <span class="option-label">{{ option }}</span>
                          </mat-radio-button>
                        </mat-radio-group>
                      </div>

                      <!-- Keyword Based -->
                      <div *ngIf="test.test_type === 'keyword_based'" class="answer-section">
                        <mat-form-field appearance="outline" class="answer-field">
                          <textarea matInput 
                                    [formControlName]="i" 
                                    rows="6" 
                                    placeholder="Введите ваш развернутый ответ здесь..."
                                    class="answer-textarea"
                                    (input)="onAnswerChange(i)"></textarea>
                        </mat-form-field>
                      </div>
                    </mat-card-content>
                  </mat-card>
                </div>
              </div>

              <!-- Navigation Controls -->
              <div class="navigation-controls">
                <button mat-icon-button 
                        type="button"
                        (click)="previousQuestion()" 
                        [disabled]="currentQuestionIndex === 0"
                        class="nav-button nav-button-left">
                  <mat-icon>chevron_left</mat-icon>
                </button>

                <div class="indicators-container">
                  <div class="indicators-wrapper" [style.transform]="getIndicatorsTransform()">
                    <div *ngFor="let question of test.questions; let i = index" 
                         class="indicator-dot"
                         [class.active]="i === currentQuestionIndex"
                         [class.answered]="hasAnswer(i)"
                         (click)="goToQuestion(i)"
                         [title]="'Вопрос ' + (i + 1)">
                    </div>
                  </div>
                </div>

                <button mat-icon-button 
                        type="button"
                        (click)="nextQuestion()" 
                        [disabled]="currentQuestionIndex === test.questions.length - 1"
                        class="nav-button nav-button-right">
                  <mat-icon>chevron_right</mat-icon>
                </button>
              </div>

              <div class="question-counter">
                Вопрос {{ currentQuestionIndex + 1 }} из {{ test.questions.length }}
              </div>

              <div class="form-actions">
                <button mat-raised-button 
                        color="primary" 
                        type="submit" 
                        [disabled]="submitting || timeExpired"
                        class="submit-button">
                  <span *ngIf="!submitting">Завершить тест</span>
                  <span *ngIf="submitting">Отправка...</span>
                </button>
                <button mat-stroked-button 
                        type="button" 
                        routerLink="/tests"
                        class="cancel-button">
                  Отмена
                </button>
              </div>
            </div>
          </form>

          <!-- Quick Navigation Sidebar -->
          <div class="quick-nav-sidebar">
            <h3 class="sidebar-title">Навигация по вопросам</h3>
            <div class="questions-list">
              <button *ngFor="let question of test.questions; let i = index"
                      type="button"
                      class="question-nav-item"
                      [class.active]="i === currentQuestionIndex"
                      [class.answered]="hasAnswer(i)"
                      (click)="goToQuestion(i)">
                <span class="question-nav-number">{{ i + 1 }}</span>
                <span class="question-nav-status" *ngIf="hasAnswer(i)">
                  <mat-icon>check_circle</mat-icon>
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .test-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .test-content {
      max-width: 1400px;
      margin: 0 auto;
    }

    .test-main-layout {
      display: flex;
      gap: 24px;
      align-items: flex-start;
    }

    .test-header {
      background: white;
      border-radius: 16px;
      padding: 32px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 24px;
    }

    .header-info {
      flex: 1;
    }

    .test-title {
      font-size: 32px;
      font-weight: 600;
      color: #1a237e;
      margin: 0 0 12px 0;
      line-height: 1.2;
    }

    .test-description {
      font-size: 16px;
      color: #616161;
      margin: 0;
      line-height: 1.6;
    }

    .timer-container {
      flex-shrink: 0;
    }

    .timer {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 12px;
      padding: 20px 24px;
      color: white;
      display: flex;
      align-items: center;
      gap: 16px;
      min-width: 200px;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
      transition: all 0.3s ease;
    }

    .timer.timer-warning {
      background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
      animation: pulse 1s infinite;
    }

    .timer.timer-critical {
      background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
      animation: pulse 0.5s infinite;
    }

    @keyframes pulse {
      0%, 100% {
        transform: scale(1);
      }
      50% {
        transform: scale(1.02);
      }
    }

    .timer-icon {
      font-size: 32px;
      width: 32px;
      height: 32px;
    }

    .timer-display {
      display: flex;
      flex-direction: column;
    }

    .timer-value {
      font-size: 28px;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      letter-spacing: 1px;
      line-height: 1;
    }

    .timer-label {
      font-size: 12px;
      opacity: 0.9;
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .test-form {
      flex: 1;
      background: white;
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }

    .questions-carousel {
      position: relative;
    }

    .carousel-wrapper {
      overflow: hidden;
      margin-bottom: 32px;
    }

    .carousel-track {
      display: flex;
      transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
      will-change: transform;
    }

    .question-card {
      min-width: 100%;
      flex-shrink: 0;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: box-shadow 0.3s ease, transform 0.3s ease;
      overflow: hidden;
      opacity: 0.3;
      transform: scale(0.95);
      pointer-events: none;
    }

    .question-card.active {
      opacity: 1;
      transform: scale(1);
      pointer-events: auto;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }

    .question-card:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }

    .question-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .question-number {
      font-size: 14px;
      font-weight: 500;
      opacity: 0.95;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .question-points {
      font-size: 14px;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.2);
      padding: 4px 12px;
      border-radius: 12px;
      text-rendering: optimizeLegibility;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      text-shadow: 0 0 0 transparent;
      opacity: 1;
    }

    .question-content {
      padding: 24px;
    }

    .question-title {
      font-size: 20px;
      font-weight: 500;
      color: #212121;
      margin: 0 0 24px 0;
      line-height: 1.5;
    }

    .answer-section {
      margin-top: 16px;
    }

    .radio-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .radio-option {
      padding: 16px;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      transition: all 0.2s ease;
      background: #fafafa;
    }

    .radio-option:hover {
      border-color: #667eea;
      background: #f3f4ff;
    }

    .radio-option ::ng-deep .mat-radio-checked .mat-radio-outer-circle {
      border-color: #667eea;
    }

    .radio-option ::ng-deep .mat-radio-checked .mat-radio-inner-circle {
      background-color: #667eea;
    }

    .option-label {
      font-size: 16px;
      color: #424242;
      margin-left: 8px;
    }

    .answer-field {
      width: 100%;
    }

    .answer-textarea {
      font-size: 16px;
      line-height: 1.6;
    }

    .navigation-controls {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 16px;
      margin: 32px 0 16px 0;
      padding: 24px;
      background: #f8f9fa;
      border-radius: 12px;
    }

    .nav-button {
      width: 48px;
      height: 48px;
      background: white;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      padding: 0;
      margin: 0;
    }

    .nav-button:hover:not(:disabled) {
      background: #667eea;
      color: white;
      transform: scale(1.1);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .nav-button:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .nav-button mat-icon {
      font-size: 32px;
      width: 32px;
      height: 32px;
      line-height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .indicators-container {
      flex: 1;
      overflow: hidden;
      max-width: 400px;
      height: 32px;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 8px;
    }

    .indicators-wrapper {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: transform 0.3s ease;
      will-change: transform;
      height: 100%;
    }

    .indicator-dot {
      width: 12px;
      height: 12px;
      min-width: 12px;
      min-height: 12px;
      border-radius: 50%;
      background: #e0e0e0;
      cursor: pointer;
      transition: all 0.3s ease;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      box-sizing: border-box;
    }

    .indicator-dot:hover {
      transform: scale(1.3);
      background: #667eea;
    }

    .indicator-dot.active {
      background: #667eea;
      transform: scale(1.4);
      box-shadow: 0 0 8px rgba(102, 126, 234, 0.5);
    }

    .indicator-dot.answered {
      background: #4caf50;
    }

    .indicator-dot.answered.active {
      background: #667eea;
    }

    .question-counter {
      text-align: center;
      font-size: 16px;
      font-weight: 500;
      color: #616161;
      margin-bottom: 24px;
    }

    .form-actions {
      display: flex;
      gap: 16px;
      margin-top: 32px;
      padding-top: 24px;
      border-top: 1px solid #e0e0e0;
      justify-content: flex-end;
    }

    .quick-nav-sidebar {
      width: 280px;
      background: white;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
      overflow-y: auto;
    }

    .sidebar-title {
      font-size: 18px;
      font-weight: 600;
      color: #1a237e;
      margin: 0 0 20px 0;
      padding-bottom: 16px;
      border-bottom: 2px solid #e0e0e0;
    }

    .questions-list {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
    }

    .question-nav-item {
      width: 44px;
      height: 44px;
      border-radius: 8px;
      border: 2px solid #e0e0e0;
      background: white;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      transition: all 0.3s ease;
      font-weight: 500;
      color: #424242;
    }

    .question-nav-item:hover {
      border-color: #667eea;
      transform: translateY(-2px);
      box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
    }

    .question-nav-item.active {
      background: #667eea;
      border-color: #667eea;
      color: white;
      transform: scale(1.1);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .question-nav-item.answered {
      background: #4caf50;
      border-color: #4caf50;
      color: white;
    }

    .question-nav-item.answered.active {
      background: #667eea;
      border-color: #667eea;
    }

    .question-nav-number {
      font-size: 14px;
    }

    .question-nav-status {
      position: absolute;
      top: -4px;
      right: -4px;
      width: 16px;
      height: 16px;
      background: white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .question-nav-status mat-icon {
      font-size: 12px;
      width: 12px;
      height: 12px;
      color: #4caf50;
    }

    .submit-button {
      min-width: 180px;
      height: 48px;
      font-size: 16px;
      font-weight: 500;
      border-radius: 8px;
    }

    .cancel-button {
      min-width: 120px;
      height: 48px;
      font-size: 16px;
      border-radius: 8px;
    }

    .error-container {
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 60vh;
    }

    .error-content {
      background: white;
      border-radius: 16px;
      padding: 48px;
      text-align: center;
      max-width: 500px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    .error-icon {
      font-size: 80px;
      width: 80px;
      height: 80px;
      color: #f44336;
      margin-bottom: 24px;
    }

    .error-content h2 {
      color: #1a237e;
      font-size: 24px;
      font-weight: 600;
      margin: 0 0 16px 0;
    }

    .error-content p {
      color: #616161;
      font-size: 16px;
      margin: 0 0 32px 0;
      line-height: 1.6;
    }

    .back-button {
      min-width: 200px;
      height: 48px;
      font-size: 16px;
      border-radius: 8px;
    }

    @media (max-width: 1200px) {
      .test-main-layout {
        flex-direction: column;
      }

      .quick-nav-sidebar {
        width: 100%;
        position: static;
        max-height: none;
      }

      .questions-list {
        grid-template-columns: repeat(10, 1fr);
      }
    }

    @media (max-width: 768px) {
      .test-container {
        padding: 16px;
      }

      .test-header {
        flex-direction: column;
        padding: 24px;
      }

      .timer-container {
        width: 100%;
      }

      .timer {
        width: 100%;
        justify-content: center;
      }

      .test-title {
        font-size: 24px;
      }

      .test-form {
        padding: 24px;
      }

      .navigation-controls {
        padding: 16px;
        gap: 8px;
      }

      .indicators-container {
        max-width: 200px;
      }

      .questions-list {
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
      }

      .question-nav-item {
        width: 36px;
        height: 36px;
      }

      .form-actions {
        flex-direction: column;
      }

      .submit-button,
      .cancel-button {
        width: 100%;
      }
    }
  `]
})
export class TestTakeComponent implements OnInit, OnDestroy {
  test: any = null;
  answerForm!: FormGroup;
  submissionId: string | null = null;
  submitting = false;
  isTestExpired: boolean = false;
  hasTimeLimit: boolean = false;
  timeLimitMinutes: number = 0;
  timeRemaining: number = 0; // в секундах
  timeExpired: boolean = false;
  timerSubscription?: Subscription;
  startTime: Date | null = null;
  currentQuestionIndex: number = 0;
  answeredQuestions: Set<number> = new Set();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private fb: FormBuilder,
    private apiService: ApiService,
    private auth: AuthService
  ) { }

  ngOnInit() {
    const testId = this.route.snapshot.paramMap.get('id');
    if (testId) {
      this.loadTest(testId);
    }
  }

  loadTest(testId: string) {
    this.apiService.getTest(testId).subscribe({
      next: (test) => {
        this.test = test;

        // Проверяем доступность теста по дедлайну
        if (test.due_date) {
          // Дата хранится в московском времени (формат +03:00)
          // Просто сравниваем напрямую
          const dueDate = new Date(test.due_date);
          const now = new Date();

          // Тест недоступен только если дедлайн уже прошел
          if (dueDate.getTime() < now.getTime()) {
            this.isTestExpired = true;
            return;
          }
        }

        // Проверяем наличие таймера
        if (test.time_limit_minutes && test.time_limit_minutes > 0) {
          this.hasTimeLimit = true;
          this.timeLimitMinutes = test.time_limit_minutes;
        }

        this.initForm();
        this.startSubmission(testId);
      },
      error: (err) => {
        console.error('Error loading test:', err);
        alert('Ошибка загрузки теста');
      }
    });
  }

  initForm() {
    const answers = this.fb.array(
      this.test.questions.map(() => this.fb.control(''))
    );
    this.answerForm = this.fb.group({ answers });

    // Проверяем начальные ответы
    this.checkAllAnswers();

    // Отслеживаем изменения в форме для определения отвеченных вопросов
    this.answerForm.valueChanges.subscribe(() => {
      this.checkAllAnswers();
    });
  }

  checkAllAnswers() {
    if (!this.test || !this.answerForm) return;

    this.test.questions.forEach((_question: any, index: number) => {
      if (this.hasAnswer(index)) {
        this.answeredQuestions.add(index);
      } else {
        this.answeredQuestions.delete(index);
      }
    });
  }

  startSubmission(testId: string) {
    const currentUser = this.auth.getCurrentUser();
    if (!currentUser) {
      alert('Сначала войдите или зарегистрируйтесь');
      this.router.navigate(['/login']);
      return;
    }
    const submissionData = {
      test_id: testId,
      user: currentUser.name,
      answers: this.test.questions.map((q: any) => ({
        question_id: q.question_id,
        answer: ''
      }))
    };

    this.apiService.createSubmission(submissionData).subscribe({
      next: (submission) => {
        this.submissionId = submission.id;
        // Используем текущее время клиента как время начала для таймера
        // Это гарантирует правильную работу независимо от часовых поясов сервера
        this.startTime = new Date();
        console.log('Submission created:', submission.id, 'Timer started at:', this.startTime, 'Time limit:', this.timeLimitMinutes, 'minutes');

        // Track test start activity
        const currentUser = this.auth.getCurrentUser();
        if (currentUser) {
          this.apiService.createActivity({
            user_name: currentUser.name,
            action_type: 'test_start',
            resource_type: 'test',
            resource_id: testId,
            session_duration: null
          }).subscribe({
            error: (err) => console.error('Error tracking test start activity:', err)
          });
        }

        // Запускаем таймер если есть ограничение по времени
        if (this.hasTimeLimit) {
          this.startTimer();
        }
      },
      error: (err) => {
        console.error('Error creating submission:', err);
        alert('Ошибка при создании сдачи теста. Попробуйте еще раз.');
      }
    });
  }

  startTimer() {
    if (!this.hasTimeLimit || !this.startTime) {
      console.warn('Cannot start timer: hasTimeLimit=', this.hasTimeLimit, 'startTime=', this.startTime);
      return;
    }

    // Вычисляем время окончания на основе текущего времени клиента
    const endTime = new Date(this.startTime.getTime() + this.timeLimitMinutes * 60 * 1000);
    const now = new Date();
    const initialRemaining = Math.max(0, Math.floor((endTime.getTime() - now.getTime()) / 1000));

    console.log('Timer initialized:', {
      startTime: this.startTime,
      endTime: endTime,
      now: now,
      timeLimitMinutes: this.timeLimitMinutes,
      initialRemaining: initialRemaining
    });

    // Если время уже истекло, сразу завершаем тест
    if (initialRemaining === 0) {
      console.error('Timer expired immediately! This should not happen.');
      this.timeRemaining = 0;
      this.timeExpired = true;
      this.autoSubmit();
      return;
    }

    this.timeRemaining = initialRemaining;

    // Обновляем таймер каждую секунду
    this.timerSubscription = interval(1000).subscribe(() => {
      const currentTime = new Date();
      const remaining = Math.max(0, Math.floor((endTime.getTime() - currentTime.getTime()) / 1000));

      this.timeRemaining = remaining;

      if (remaining === 0 && !this.timeExpired) {
        this.timeExpired = true;
        if (this.timerSubscription) {
          this.timerSubscription.unsubscribe();
        }
        this.autoSubmit();
      }
    });
  }

  formatTime(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  previousQuestion() {
    if (this.currentQuestionIndex > 0) {
      this.currentQuestionIndex--;
    }
  }

  nextQuestion() {
    if (this.currentQuestionIndex < this.test.questions.length - 1) {
      this.currentQuestionIndex++;
    }
  }

  goToQuestion(index: number) {
    if (index >= 0 && index < this.test.questions.length) {
      this.currentQuestionIndex = index;
    }
  }

  hasAnswer(index: number): boolean {
    if (!this.answerForm) return false;
    const answer = this.answerForm.value.answers[index];
    return answer && answer.toString().trim() !== '';
  }

  onAnswerChange(index: number) {
    if (this.hasAnswer(index)) {
      this.answeredQuestions.add(index);
    } else {
      this.answeredQuestions.delete(index);
    }
  }

  getIndicatorsTransform(): string {
    const maxVisible = 12; // Максимум видимых индикаторов
    const total = this.test?.questions?.length || 0;

    if (total <= maxVisible) {
      return 'translateX(0)';
    }

    // Вычисляем смещение так, чтобы текущий вопрос был в центре видимой области
    const current = this.currentQuestionIndex;
    const halfVisible = Math.floor(maxVisible / 2);
    let offset = current - halfVisible;

    // Ограничиваем смещение
    if (offset < 0) {
      offset = 0;
    } else if (offset > total - maxVisible) {
      offset = total - maxVisible;
    }

    // Смещаем на отрицательное значение (12px ширина + 8px gap = 20px на индикатор)
    return `translateX(-${offset * 20}px)`;
  }

  autoSubmit() {
    // Защита от множественных вызовов
    if (this.submitting) return;

    this.submitting = true;
    this.timeExpired = true;

    // Сохраняем текущие ответы перед автоматической отправкой
    if (this.answerForm && this.submissionId && this.test) {
      const answers = this.answerForm.value.answers.map((answer: string, index: number) => ({
        question_id: this.test.questions[index].question_id,
        answer: answer || ''
      }));

      this.apiService.updateSubmission(this.submissionId, { answers }).subscribe({
        next: () => {
          this.apiService.finishSubmission(this.submissionId!, false).subscribe({
            next: (result) => {
              // Track test finish activity (auto-submit)
              const currentUser = this.auth.getCurrentUser();
              if (currentUser && this.test) {
                const duration = this.startTime ? Math.floor((new Date().getTime() - this.startTime.getTime()) / 1000) : null;
                this.apiService.createActivity({
                  user_name: currentUser.name,
                  action_type: 'test_finish',
                  resource_type: 'test',
                  resource_id: this.test.id,
                  session_duration: duration
                }).subscribe({
                  error: (err) => console.error('Error tracking auto-submit activity:', err)
                });
              }

              alert('Время на прохождение теста истекло. Тест автоматически завершен.');
              this.router.navigate(['/submissions', this.submissionId, 'results']);
            },
            error: (err) => {
              console.error('Error finishing submission:', err);
              alert('Время истекло, но произошла ошибка при завершении теста. Ваши ответы сохранены.');
              this.submitting = false;
            }
          });
        },
        error: (err) => {
          console.error('Error updating submission:', err);
          alert('Время истекло, но произошла ошибка при сохранении ответов.');
          this.submitting = false;
        }
      });
    } else {
      this.submitting = false;
    }
  }

  onSubmit() {
    if (!this.submissionId) {
      alert('Ошибка: сдача теста не создана. Попробуйте обновить страницу.');
      return;
    }

    if (!this.answerForm) {
      alert('Ошибка: форма не инициализирована.');
      return;
    }

    if (this.timeExpired) {
      return;
    }

    this.submitting = true;

    const answers = this.answerForm.value.answers.map((answer: string, index: number) => ({
      question_id: this.test.questions[index].question_id,
      answer: answer || ''
    }));

    this.apiService.updateSubmission(this.submissionId, { answers }).subscribe({
      next: () => {
        this.apiService.finishSubmission(this.submissionId!, false).subscribe({
          next: (result) => {
            // Track test finish activity
            const currentUser = this.auth.getCurrentUser();
            if (currentUser && this.test) {
              const duration = this.startTime ? Math.floor((new Date().getTime() - this.startTime.getTime()) / 1000) : null;
              this.apiService.createActivity({
                user_name: currentUser.name,
                action_type: 'test_finish',
                resource_type: 'test',
                resource_id: this.test.id,
                session_duration: duration
              }).subscribe({
                error: (err) => console.error('Error tracking test finish activity:', err)
              });
            }

            // Redirect to course page if subject_id is available
            if (this.test && this.test.subject_id) {
              alert('Тест завершен! Ваш результат сохранен.');
              this.router.navigate(['/courses', this.test.subject_id]);
            } else {
              this.router.navigate(['/submissions', this.submissionId, 'results']);
            }
          },
          error: (err) => {
            console.error('Error finishing submission:', err);
            alert('Ошибка при завершении теста: ' + (err.error?.detail || err.message || 'Неизвестная ошибка'));
            this.submitting = false;
          }
        });
      },
      error: (err) => {
        console.error('Error updating submission:', err);
        alert('Ошибка при сохранении ответов: ' + (err.error?.detail || err.message || 'Неизвестная ошибка'));
        this.submitting = false;
      }
    });
  }

  ngOnDestroy() {
    if (this.timerSubscription) {
      this.timerSubscription.unsubscribe();
    }
  }
}

