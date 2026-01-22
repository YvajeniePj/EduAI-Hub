import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatListModule } from '@angular/material/list';
import { MatChipsModule } from '@angular/material/chips';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-submission-results',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule,
    MatListModule,
    MatChipsModule,
    MatExpansionModule,
    MatProgressSpinnerModule
  ],
  template: `
    <div class="results-container" *ngIf="results">
      <div class="results-content">
        <div class="results-header">
          <h1 class="results-title">Результаты теста</h1>
        </div>
        
        <mat-card class="summary-card">
          <mat-card-content class="summary-content">
            <div class="summary-info">
              <h2 class="summary-label">Итоговая оценка</h2>
              <div class="score-display">
                <span class="score-value">{{ results.submission.total_score }}</span>
                <span class="score-separator">/</span>
                <span class="score-max">{{ results.submission.total_max }}</span>
              </div>
              <div class="points-info">
                <span class="points-label">Начислено очков:</span>
                <span class="points-value">{{ results.submission.points_awarded }}</span>
              </div>
            </div>
          </mat-card-content>
        </mat-card>

        <div class="questions-section">
          <h2 class="section-title">Детали по вопросам</h2>
          <mat-card *ngFor="let result of results.per_question_results; let i = index" class="result-card">
            <mat-card-header class="result-header">
              <div class="question-number">Вопрос {{ i + 1 }} из {{ results.per_question_results.length }}</div>
              <div class="question-score" [class.score-full]="result.score === result.max_points" 
                   [class.score-partial]="result.score > 0 && result.score < result.max_points"
                   [class.score-zero]="result.score === 0">
                {{ result.score }} / {{ result.max_points }}
              </div>
            </mat-card-header>
            <mat-card-content class="result-content">
              <h3 class="question-title">{{ result.title }}</h3>
              
              <div class="answer-section">
                <div class="answer-label">Ваш ответ:</div>
                <div class="answer-text">{{ result.answer || 'Ответ не предоставлен' }}</div>
              </div>
              
              <div *ngIf="result.details && result.details.length > 0" class="details-section">
                <div class="section-label">Детали оценки:</div>
                <ul class="details-list">
                  <li *ngFor="let detail of result.details" class="detail-item">{{ detail }}</li>
                </ul>
              </div>

              <!-- AI Feedback for keyword-based tests -->
              <div *ngIf="testType === 'keyword_based' && result.ai_feedback" class="ai-feedback-section">
                <div class="section-label">AI-оценка:</div>
                <div class="ai-feedback-content">
                  <div *ngIf="result.ai_feedback.recommended_score !== undefined" class="feedback-item">
                    <span class="feedback-label">Рекомендованный балл:</span>
                    <span class="feedback-value">{{ result.ai_feedback.recommended_score }} / {{ result.max_points }}</span>
                  </div>
                  <div *ngIf="result.ai_feedback.found_keywords && result.ai_feedback.found_keywords.length > 0" class="feedback-item">
                    <div class="feedback-label">Найденные ключевые слова:</div>
                    <div class="chips-container">
                      <mat-chip *ngFor="let kw of result.ai_feedback.found_keywords" class="keyword-chip found">{{ kw }}</mat-chip>
                    </div>
                  </div>
                  <div *ngIf="result.ai_feedback.missing_keywords && result.ai_feedback.missing_keywords.length > 0" class="feedback-item">
                    <div class="feedback-label">Отсутствующие ключевые слова:</div>
                    <div class="chips-container">
                      <mat-chip *ngFor="let kw of result.ai_feedback.missing_keywords" class="keyword-chip missing">{{ kw }}</mat-chip>
                    </div>
                  </div>
                  <div *ngIf="result.ai_feedback.evaluation" class="feedback-item">
                    <span class="feedback-label">Оценка:</span>
                    <span class="feedback-value">{{ result.ai_feedback.evaluation }}</span>
                  </div>
                  <div *ngIf="result.ai_feedback.feedback" class="feedback-item">
                    <div class="feedback-label">Обратная связь:</div>
                    <div class="feedback-text">{{ result.ai_feedback.feedback }}</div>
                  </div>
                </div>
              </div>

              <!-- AI Feedback for multiple choice AI-generated tests -->
              <div *ngIf="testType === 'multiple_choice' && isAiGenerated && result.score < result.max_points" class="ai-materials-section">
                <mat-expansion-panel *ngIf="result.aiFeedback" class="materials-panel">
                  <mat-expansion-panel-header class="materials-panel-header">
                    <mat-panel-title class="materials-panel-title">
                      Показать ответ из материалов
                    </mat-panel-title>
                  </mat-expansion-panel-header>
                  <div class="materials-content">
                    <div *ngIf="result.aiFeedback.materials_info && result.aiFeedback.materials_info.length > 0" class="materials-list-section">
                      <div class="section-label">Материалы, по которым был создан тест:</div>
                      <ul class="materials-list">
                        <li *ngFor="let material of result.aiFeedback.materials_info" class="material-item">
                          {{ material.original_name || material.name }}
                        </li>
                      </ul>
                    </div>
                    <div *ngIf="result.aiFeedback.material_answers && result.aiFeedback.material_answers.length > 0" class="materials-answers-section">
                      <div class="section-label">Ответы из материалов:</div>
                      <div *ngFor="let materialAnswer of result.aiFeedback.material_answers" class="material-answer-item">
                        <div class="material-name">{{ materialAnswer.material_name }}:</div>
                        <div class="material-answer-text">{{ materialAnswer.answer }}</div>
                      </div>
                    </div>
                    <div *ngIf="!result.aiFeedback.material_answers || result.aiFeedback.material_answers.length === 0" class="no-materials">
                      Информация из материалов недоступна
                    </div>
                  </div>
                </mat-expansion-panel>
                <div *ngIf="!result.aiFeedback && !loadingFeedback[i]" class="load-feedback-button">
                  <button mat-stroked-button (click)="loadAiFeedback(i, result)" [disabled]="loadingFeedback[i]">
                    Загрузить ответ из материалов
                  </button>
                </div>
                <div *ngIf="loadingFeedback[i]" class="loading-feedback">
                  <mat-spinner diameter="30"></mat-spinner>
                </div>
              </div>
            </mat-card-content>
          </mat-card>
        </div>

        <div class="actions">
          <button mat-raised-button color="primary" routerLink="/tests" class="back-button">
            Вернуться к тестам
          </button>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .results-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .results-content {
      max-width: 1000px;
      margin: 0 auto;
    }

    .results-header {
      margin-bottom: 24px;
    }

    .results-title {
      font-size: 32px;
      font-weight: 600;
      color: #1a237e;
      margin: 0;
    }

    .summary-card {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      border-radius: 16px;
      margin-bottom: 32px;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .summary-content {
      padding: 32px;
    }

    .summary-info {
      text-align: center;
      color: white;
    }

    .summary-label {
      font-size: 18px;
      font-weight: 500;
      margin: 0 0 16px 0;
      opacity: 0.95;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .score-display {
      display: flex;
      align-items: baseline;
      justify-content: center;
      gap: 8px;
      margin-bottom: 16px;
    }

    .score-value {
      font-size: 56px;
      font-weight: 700;
      line-height: 1;
      text-rendering: optimizeLegibility;
      -webkit-font-smoothing: antialiased;
    }

    .score-separator {
      font-size: 32px;
      font-weight: 400;
      opacity: 0.8;
    }

    .score-max {
      font-size: 32px;
      font-weight: 500;
      opacity: 0.9;
    }

    .points-info {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-size: 16px;
      opacity: 0.95;
    }

    .points-label {
      font-weight: 400;
    }

    .points-value {
      font-weight: 600;
      font-size: 18px;
    }

    .questions-section {
      margin-bottom: 32px;
    }

    .section-title {
      font-size: 24px;
      font-weight: 600;
      color: #1a237e;
      margin: 0 0 24px 0;
    }

    .result-card {
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      margin-bottom: 24px;
      overflow: hidden;
      transition: box-shadow 0.3s ease;
    }

    .result-card:hover {
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }

    .result-header {
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

    .question-score {
      font-size: 16px;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.2);
      padding: 6px 16px;
      border-radius: 12px;
      text-rendering: optimizeLegibility;
      -webkit-font-smoothing: antialiased;
    }

    .question-score.score-full {
      background: rgba(76, 175, 80, 0.3);
    }

    .question-score.score-partial {
      background: rgba(255, 193, 7, 0.3);
    }

    .question-score.score-zero {
      background: rgba(244, 67, 54, 0.3);
    }

    .result-content {
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
      margin-bottom: 24px;
      padding: 16px;
      background: #f8f9fa;
      border-radius: 8px;
      border-left: 4px solid #667eea;
    }

    .answer-label {
      font-size: 14px;
      font-weight: 600;
      color: #616161;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }

    .answer-text {
      font-size: 16px;
      color: #212121;
      line-height: 1.6;
      white-space: pre-wrap;
    }

    .details-section,
    .ai-feedback-section,
    .ai-materials-section {
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid #e0e0e0;
    }

    .section-label {
      font-size: 16px;
      font-weight: 600;
      color: #424242;
      margin-bottom: 12px;
    }

    .details-list {
      margin: 0;
      padding-left: 24px;
      list-style-type: disc;
    }

    .detail-item {
      font-size: 16px;
      color: #424242;
      line-height: 1.8;
      margin-bottom: 8px;
    }

    .ai-feedback-content {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .feedback-item {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .feedback-label {
      font-size: 14px;
      font-weight: 600;
      color: #616161;
    }

    .feedback-value {
      font-size: 16px;
      color: #212121;
      font-weight: 500;
    }

    .feedback-text {
      font-size: 16px;
      color: #424242;
      line-height: 1.6;
      white-space: pre-wrap;
    }

    .chips-container {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .keyword-chip {
      font-size: 14px;
    }

    .keyword-chip.found {
      background: #4caf50;
      color: white;
    }

    .keyword-chip.missing {
      background: #f44336;
      color: white;
    }

    .materials-panel {
      margin-top: 16px;
    }

    .materials-panel-header {
      background: #f8f9fa;
    }

    .materials-panel-title {
      font-size: 16px;
      font-weight: 500;
      color: #667eea;
    }

    .materials-content {
      padding: 16px 0;
    }

    .materials-list-section {
      margin-bottom: 24px;
    }

    .materials-list {
      margin: 12px 0 0 0;
      padding-left: 24px;
      list-style-type: disc;
    }

    .material-item {
      font-size: 16px;
      color: #424242;
      line-height: 1.8;
      margin-bottom: 8px;
    }

    .materials-answers-section {
      margin-top: 24px;
    }

    .material-answer-item {
      margin-bottom: 20px;
      padding: 16px;
      background: #f8f9fa;
      border-radius: 8px;
      border-left: 4px solid #4caf50;
    }

    .material-name {
      font-size: 14px;
      font-weight: 600;
      color: #616161;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .material-answer-text {
      font-size: 16px;
      color: #212121;
      line-height: 1.6;
      font-style: italic;
      white-space: pre-wrap;
    }

    .no-materials {
      font-size: 16px;
      color: #616161;
      font-style: italic;
      padding: 16px;
      background: #f8f9fa;
      border-radius: 8px;
      text-align: center;
    }

    .load-feedback-button {
      margin-top: 16px;
    }

    .loading-feedback {
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }

    .actions {
      display: flex;
      justify-content: center;
      margin-top: 32px;
      padding-top: 24px;
    }

    .back-button {
      min-width: 200px;
      height: 48px;
      font-size: 16px;
      font-weight: 500;
      border-radius: 8px;
    }

    @media (max-width: 768px) {
      .results-container {
        padding: 16px;
      }

      .results-title {
        font-size: 24px;
      }

      .score-value {
        font-size: 42px;
      }

      .score-max {
        font-size: 24px;
      }

      .summary-content {
        padding: 24px;
      }

      .result-content {
        padding: 20px;
      }
    }
  `]
})
export class SubmissionResultsComponent implements OnInit {
  results: any = null;
  testType: string = '';
  isAiGenerated: boolean = false;
  materialIds: string[] = [];
  loadingFeedback: boolean[] = [];

  constructor(
    private route: ActivatedRoute,
    private apiService: ApiService
  ) {}

  ngOnInit() {
    const submissionId = this.route.snapshot.paramMap.get('id');
    if (submissionId) {
      this.loadResults(submissionId);
    }
  }

  loadResults(submissionId: string) {
    this.apiService.getSubmissionResults(submissionId).subscribe({
      next: (results) => {
        this.results = results;
        // Extract test info
        if (results.submission && results.submission.test_id) {
          this.apiService.getTest(results.submission.test_id).subscribe({
            next: (test) => {
              this.testType = test.test_type || '';
              this.isAiGenerated = test.ai_generated === 'true' || test.ai_generated === true;
              
              // Extract material IDs from description
              if (test.description) {
                const match = test.description.match(/\[AI_MATERIALS:(.+?)\]/);
                if (match) {
                  this.materialIds = match[1].split(',').map((id: string) => id.trim());
                }
              }
              
              // Initialize loadingFeedback array
              this.loadingFeedback = new Array(results.per_question_results.length).fill(false);
            },
            error: (err) => console.error('Error loading test:', err)
          });
        }
      },
      error: (err) => {
        console.error('Error loading results:', err);
        alert('Ошибка загрузки результатов');
      }
    });
  }

  loadAiFeedback(index: number, result: any) {
    if (!this.materialIds || this.materialIds.length === 0) {
      alert('Материалы для этого теста недоступны');
      return;
    }

    this.loadingFeedback[index] = true;
    
    // Get test to find correct answer
    if (this.results.submission && this.results.submission.test_id) {
      this.apiService.getTest(this.results.submission.test_id).subscribe({
        next: (test) => {
          const question = test.questions.find((q: any) => q.question_id === result.question_id);
          const correctAnswer = question?.correct_answer || '';
          
          this.apiService.getTestFeedback({
            test_id: this.results.submission.test_id,
            test_type: this.testType,
            question_id: result.question_id,
            question_title: result.title,
            student_answer: result.answer,
            correct_answer: correctAnswer,
            material_ids: this.materialIds,
            max_points: result.max_points
          }).subscribe({
            next: (feedback) => {
              if (!this.results.per_question_results[index].aiFeedback) {
                this.results.per_question_results[index].aiFeedback = feedback.feedback;
              }
              this.loadingFeedback[index] = false;
            },
            error: (err) => {
              console.error('Error loading AI feedback:', err);
              alert('Ошибка загрузки обратной связи');
              this.loadingFeedback[index] = false;
            }
          });
        },
        error: (err) => {
          console.error('Error loading test:', err);
          this.loadingFeedback[index] = false;
        }
      });
    }
  }
}

