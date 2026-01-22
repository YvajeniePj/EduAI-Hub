import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-test-detail',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule
  ],
  template: `
    <div class="container" *ngIf="test">
      <h1>{{ test.title }}</h1>
      <p>{{ test.description }}</p>
      <p><strong>Тип теста:</strong> {{ getTestTypeLabel(test.test_type) }}</p>
      <p *ngIf="test.due_date"><strong>Дедлайн:</strong> {{ test.due_date | date }}</p>

      <div class="actions">
        <button mat-raised-button color="primary" [routerLink]="['/tests', test.id, 'take']">
          Пройти тест
        </button>
        <button mat-button routerLink="/tests">Назад</button>
      </div>

      <h2>Вопросы ({{ test.questions?.length || 0 }})</h2>
      <mat-card *ngFor="let question of test.questions" class="question-card">
        <mat-card-content>
          <h3>{{ question.title }}</h3>
          <p>Максимум баллов: {{ question.max_points }}</p>
          
          <div *ngIf="test.test_type === 'multiple_choice'">
            <h4>Варианты ответов:</h4>
            <ul>
              <li *ngFor="let option of question.options">{{ option }}</li>
            </ul>
            <p><strong>Правильный ответ:</strong> {{ question.correct_answer }}</p>
          </div>

          <div *ngIf="test.test_type === 'keyword_based'">
            <h4>Ключевые слова:</h4>
            <ul>
              <li *ngFor="let keyword of question.keywords">
                {{ keyword.word }} ({{ keyword.points }} баллов)
              </li>
            </ul>
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [`
    .container {
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
    }
    .actions {
      margin: 20px 0;
    }
    .question-card {
      margin-bottom: 20px;
    }
  `]
})
export class TestDetailComponent implements OnInit {
  test: any = null;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private apiService: ApiService
  ) {}

  ngOnInit() {
    const id = this.route.snapshot.paramMap.get('id');
    if (id) {
      this.apiService.getTest(id).subscribe({
        next: (test) => this.test = test,
        error: (err) => {
          console.error('Error loading test:', err);
          alert('Ошибка загрузки теста');
          this.router.navigate(['/tests']);
        }
      });
    }
  }

  getTestTypeLabel(type: string): string {
    return type === 'multiple_choice' ? 'С вариантами ответов' : 'С ключевыми словами';
  }
}

