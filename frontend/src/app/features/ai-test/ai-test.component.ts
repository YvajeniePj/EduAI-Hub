import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatIconModule } from '@angular/material/icon';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-ai-test',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatButtonModule,
    MatInputModule,
    MatSelectModule,
    MatCheckboxModule,
    MatProgressSpinnerModule,
    MatFormFieldModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatIconModule
  ],
  template: `
    <div class="ai-test-container">
      <div class="page-header">
        <h1 class="page-title">AI-генерация тестов</h1>
        <p class="page-subtitle">Создавайте тесты автоматически с помощью искусственного интеллекта</p>
      </div>
      
      <div class="ai-test-content">
      <mat-card>
        <mat-card-content>
          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Выберите предмет</mat-label>
            <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="loadMaterials()">
              <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                {{ subject.name }}
              </mat-option>
            </mat-select>
          </mat-form-field>

          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Название теста</mat-label>
            <input matInput [(ngModel)]="testTitle" placeholder="Введите название теста">
          </mat-form-field>

          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Описание теста</mat-label>
            <textarea matInput [(ngModel)]="testDescription" rows="3" placeholder="Описание теста (опционально)"></textarea>
          </mat-form-field>

          <mat-form-field appearance="outline">
            <mat-label>Количество вопросов</mat-label>
            <input matInput type="number" [(ngModel)]="questionCount" min="1" max="20" value="5">
          </mat-form-field>

          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Тип теста</mat-label>
            <mat-select [(ngModel)]="testType">
              <mat-option value="multiple_choice">С вариантами ответов</mat-option>
              <mat-option value="keyword_based">С ключевыми словами</mat-option>
            </mat-select>
          </mat-form-field>

          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Дополнительные условия для нейросети (опционально)</mat-label>
            <textarea matInput [(ngModel)]="additionalConditions" rows="4" 
              placeholder="Например: Вопросы должны быть на русском языке, фокус на практическом применении..."></textarea>
            <mat-hint>Эти условия будут добавлены к базовому промпту для генерации теста</mat-hint>
          </mat-form-field>

          <div class="deadline-row">
            <mat-form-field appearance="outline" class="deadline-date-field">
              <mat-label>Дедлайн - Дата</mat-label>
              <input matInput [matDatepicker]="picker" [(ngModel)]="dueDate" [min]="minDate" placeholder="Выберите дату" (dateChange)="onDateChange()">
              <mat-datepicker-toggle matSuffix [for]="picker"></mat-datepicker-toggle>
              <mat-datepicker #picker startView="month"></mat-datepicker>
              <mat-hint>Дата до которой нужно сдать работу</mat-hint>
            </mat-form-field>

            <mat-form-field appearance="outline" class="deadline-time-field">
              <mat-label>Время</mat-label>
              <input matInput type="time" [(ngModel)]="dueTime" [min]="getMinTime()" placeholder="ЧЧ:ММ" [disabled]="!dueDate" (input)="onTimeChange()" (blur)="onTimeBlur()">
              <mat-icon matSuffix>schedule</mat-icon>
              <mat-hint>Время дедлайна</mat-hint>
              <mat-error *ngIf="timeError && timeFieldTouched">
                {{ timeError }}
              </mat-error>
            </mat-form-field>
          </div>

          <mat-form-field appearance="outline" style="width: 100%; margin-top: 16px;">
            <mat-label>Ограничение по времени (минуты)</mat-label>
            <input matInput type="number" [(ngModel)]="timeLimitMinutes" min="1" placeholder="Оставьте пустым, если ограничения нет">
            <mat-icon matSuffix>timer</mat-icon>
            <mat-hint>Максимальное время на прохождение теста в минутах (опционально)</mat-hint>
          </mat-form-field>

          <div *ngIf="materials.length > 0" style="margin-top: 20px;">
            <h3>Выберите материалы для генерации вопросов:</h3>
            <div *ngFor="let material of materials" style="margin-bottom: 10px;">
              <mat-checkbox [(ngModel)]="material.selected">
                {{ material.original_name || material.name }}
                <span *ngIf="material.note"> - {{ material.note }}</span>
              </mat-checkbox>
            </div>
          </div>

          <div *ngIf="materials.length === 0 && selectedSubjectId" class="info-message">
            Нет доступных материалов для генерации. Загрузите материалы во вкладке "Материалы".
          </div>

          <button 
            mat-raised-button 
            color="primary" 
            (click)="generateTest()" 
            [disabled]="!canGenerate() || generating"
            style="margin-top: 20px;">
            <mat-spinner *ngIf="generating" diameter="20" style="display: inline-block; margin-right: 10px;"></mat-spinner>
            {{ generating ? 'Генерация...' : 'Сгенерировать тест' }}
          </button>
        </mat-card-content>
      </mat-card>

      <mat-card *ngIf="generatedTest">
        <mat-card-header>
          <mat-card-title>Сгенерированный тест</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div *ngFor="let question of generatedTest.questions; let i = index" style="margin-bottom: 20px;">
            <h4>Вопрос {{ i + 1 }}: {{ question.title }}</h4>
            <p><strong>Баллы:</strong> {{ question.max_points }}</p>
            
            <!-- Multiple Choice -->
            <div *ngIf="generatedTest.test_type === 'multiple_choice'">
              <p><strong>Варианты ответов:</strong></p>
              <ul>
                <li *ngFor="let option of question.options">{{ option }}</li>
              </ul>
              <p><strong>Правильный ответ:</strong> {{ question.correct_answer }}</p>
            </div>
            
            <!-- Keyword Based -->
            <div *ngIf="generatedTest.test_type === 'keyword_based'">
              <p><strong>Ключевые слова:</strong></p>
              <ul>
                <li *ngFor="let keyword of question.keywords">
                  "{{ keyword.word }}" - {{ keyword.points }} баллов
                </li>
              </ul>
            </div>
          </div>
          
          <button mat-raised-button color="primary" (click)="saveTest()" [disabled]="saving">
            {{ saving ? 'Сохранение...' : 'Сохранить тест' }}
          </button>
        </mat-card-content>
      </mat-card>
      </div>
    </div>
  `,
  styles: [`
    .ai-test-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .ai-test-content {
      max-width: 1000px;
      margin: 0 auto;
    }

    .page-header {
      margin-bottom: 32px;
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

    mat-card {
      margin-bottom: 24px;
      border-radius: 16px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      background: white;
    }

    mat-card-content {
      padding: 24px;
    }

    .full-width {
      width: 100%;
    }

    mat-form-field {
      margin-bottom: 16px;
    }

    .deadline-row {
      display: flex;
      gap: 16px;
      align-items: flex-start;
      margin-top: 16px;
    }

    .deadline-date-field {
      flex: 1.5;
    }

    .deadline-time-field {
      flex: 1;
    }

    .info-message {
      padding: 20px;
      text-align: center;
      color: #616161;
      background: #f8f9fa;
      border-radius: 8px;
      margin-top: 20px;
      border-left: 4px solid #667eea;
    }

    button[mat-raised-button] {
      padding: 12px 32px;
      font-size: 16px;
      font-weight: 500;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
      margin-top: 24px;
    }

    @media (max-width: 768px) {
      .ai-test-container {
        padding: 16px;
      }

      .page-title {
        font-size: 24px;
      }

      .deadline-row {
        flex-direction: column;
      }

      .deadline-date-field,
      .deadline-time-field {
        width: 100%;
      }
    }
  `]
})
export class AiTestComponent implements OnInit {
  subjects: any[] = [];
  materials: any[] = [];
  selectedSubjectId: string = '';
  testTitle: string = '';
  testDescription: string = '';
  questionCount: number = 5;
  testType: string = 'multiple_choice';
  additionalConditions: string = '';
  generating: boolean = false;
  saving: boolean = false;
  generatedTest: any = null;
  dueDate: Date | null = null;
  dueTime: string = '';
  timeLimitMinutes: number | null = null;
  minDate: Date = new Date();
  timeError: string = '';
  timeFieldTouched: boolean = false;

  constructor(
    private apiService: ApiService,
    private router: Router
  ) {}

  ngOnInit() {
    this.loadSubjects();
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
        if (subjects.length > 0 && !this.selectedSubjectId) {
          this.selectedSubjectId = subjects[0].id;
          this.loadMaterials();
        }
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  loadMaterials() {
    if (!this.selectedSubjectId) return;
    
    this.apiService.getMaterials(this.selectedSubjectId).subscribe({
      next: (materials) => {
        this.materials = materials.map(m => ({ ...m, selected: false }));
      },
      error: (err) => {
        console.error('Error loading materials:', err);
        this.materials = [];
      }
    });
  }

  canGenerate(): boolean {
    return !!(
      this.selectedSubjectId &&
      this.testTitle.trim() &&
      this.questionCount > 0 &&
      this.materials.some(m => m.selected) &&
      this.testType
    );
  }

  generateTest() {
    if (!this.canGenerate()) return;

    const selectedMaterials = this.materials
      .filter(m => m.selected)
      .map(m => m.id);

    this.generating = true;
    this.generatedTest = null;

    this.apiService.generateTest({
      title: this.testTitle,
      subject_id: this.selectedSubjectId,
      description: this.testDescription,
      question_count: this.questionCount,
      material_ids: selectedMaterials,
      test_type: this.testType,
      additional_conditions: this.additionalConditions.trim() || undefined
    }).subscribe({
      next: (result) => {
        this.generating = false;
        this.generatedTest = result;
      },
      error: (err) => {
        this.generating = false;
        console.error('Error generating test:', err);
        alert('Ошибка при генерации теста: ' + (err.error?.detail || err.message));
      }
    });
  }

  getMinTime(): string {
    if (!this.dueDate) {
      const now = new Date();
      return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    }
    const today = new Date();
    const selectedDate = new Date(this.dueDate);
    if (selectedDate.toDateString() === today.toDateString()) {
      const now = new Date();
      return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    }
    return '00:00';
  }

  validateTime(): boolean {
    if (!this.dueDate || !this.dueTime) {
      this.timeError = '';
      return true; // Если дата или время не выбраны, валидация не требуется
    }

    const [hours, minutes] = this.dueTime.split(':');
    const deadline = new Date(this.dueDate);
    deadline.setHours(parseInt(hours), parseInt(minutes), 0, 0);
    
    const now = new Date();
    
    if (deadline.getTime() < now.getTime()) {
      this.timeError = 'Выбранное время уже прошло. Выберите будущее время.';
      return false; // Время в прошлом
    }
    
    this.timeError = '';
    return true; // Время валидно
  }

  onDateChange() {
    this.timeFieldTouched = true; // Помечаем как touched при изменении даты
    if (this.dueTime) {
      this.validateTime();
    } else {
      this.timeError = '';
    }
  }

  onTimeChange() {
    this.timeFieldTouched = true; // Помечаем как touched при изменении времени
    this.validateTime();
  }

  onTimeBlur() {
    this.timeFieldTouched = true;
    if (this.dueTime) {
      this.validateTime();
    }
  }

  saveTest() {
    if (!this.generatedTest || !this.selectedSubjectId) return;

    // Проверяем валидность времени дедлайна
    if (this.dueDate && this.dueTime) {
      if (!this.validateTime()) {
        alert('Невозможно сохранить тест: выбранное время дедлайна уже прошло. Выберите будущее время.');
        return;
      }
    }

    this.saving = true;

    const testType = this.generatedTest.test_type || 'multiple_choice';
    const materialIds = this.generatedTest.material_ids || [];
    
    // Формируем дату дедлайна если указана
    let dueDateStr: string | null = null;
    if (this.dueDate && this.dueTime) {
      const [hours, minutes] = this.dueTime.split(':');
      const deadline = new Date(this.dueDate);
      deadline.setHours(parseInt(hours), parseInt(minutes), 0, 0);
      // Форматируем как московское время
      const year = deadline.getFullYear();
      const month = String(deadline.getMonth() + 1).padStart(2, '0');
      const day = String(deadline.getDate()).padStart(2, '0');
      dueDateStr = `${year}-${month}-${day}T${hours}:${minutes}:00+03:00`;
    }
    
    const testData: any = {
      subject_id: this.selectedSubjectId,
      title: this.testTitle,
      description: this.testDescription,
      test_type: testType,
      ai_generated: true,
      due_date: dueDateStr,
      available_until: dueDateStr,
      time_limit_minutes: this.timeLimitMinutes || null,
      questions: this.generatedTest.questions.map((q: any, index: number) => {
        const question: any = {
          question_id: 'q' + (index + 1),
          title: q.title,
          max_points: q.max_points || 10,
          test_type: testType
        };
        
        if (testType === 'multiple_choice') {
          question.options = q.options || [];
          question.correct_answer = q.correct_answer || '';
        } else if (testType === 'keyword_based') {
          question.keywords = q.keywords || [];
        }
        
        return question;
      })
    };
    
    // Store material_ids in test metadata for feedback (we'll need to extend Test model or use description)
    if (materialIds.length > 0) {
      testData.description = (testData.description || '') + `\n[AI_MATERIALS:${materialIds.join(',')}]`;
    }

    this.apiService.createTest(testData).subscribe({
      next: () => {
        this.saving = false;
        alert('Тест успешно создан!');
        this.router.navigate(['/tests']);
      },
      error: (err) => {
        this.saving = false;
        console.error('Error saving test:', err);
        alert('Ошибка при сохранении теста: ' + (err.error?.detail || err.message));
      }
    });
  }
}

