import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { ApiService } from '../../core/services/api.service';
import { RussianDatePipe } from '../../core/pipes/russian-date.pipe';

@Component({
  selector: 'app-news-manage',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatIconModule,
    MatChipsModule,
    RussianDatePipe
  ],
  template: `
    <div class="news-manage-container">
      <h1>Управление новостями</h1>

      <!-- Форма создания новости -->
      <mat-card>
        <mat-card-header>
          <mat-card-title>Создать новость</mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <form [formGroup]="newsForm" (ngSubmit)="onSubmit()">
            <mat-form-field appearance="outline" style="width: 100%;">
              <mat-label>Курс</mat-label>
              <mat-select formControlName="subject_id" required>
                <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                  {{ subject.name }}
                </mat-option>
              </mat-select>
            </mat-form-field>

            <mat-form-field appearance="outline" style="width: 100%;">
              <mat-label>Заголовок</mat-label>
              <input matInput formControlName="title" required>
            </mat-form-field>

            <mat-form-field appearance="outline" style="width: 100%;">
              <mat-label>Содержание</mat-label>
              <textarea matInput formControlName="content" rows="6" required></textarea>
            </mat-form-field>

            <mat-form-field appearance="outline" style="width: 100%;">
              <mat-label>URL изображения (опционально)</mat-label>
              <input matInput formControlName="image_url" placeholder="https://example.com/image.jpg">
              <mat-hint>Введите URL изображения для новости</mat-hint>
            </mat-form-field>

            <div class="actions">
              <button mat-raised-button color="primary" type="submit" [disabled]="!newsForm.valid || submitting">
                {{ editingNews ? 'Обновить' : 'Создать' }} новость
              </button>
              <button mat-button type="button" (click)="resetForm()" *ngIf="editingNews">
                Отмена
              </button>
            </div>
          </form>
        </mat-card-content>
      </mat-card>

      <!-- Список новостей -->
      <h2 style="margin-top: 40px;">Все новости</h2>
      <div *ngIf="allNews.length === 0" class="empty-state">
        <p>Новостей пока нет</p>
      </div>
      <div class="news-list">
        <mat-card *ngFor="let news of allNews" class="news-item">
          <img *ngIf="news.image_url" [src]="news.image_url" alt="News image" class="news-image">
          <mat-card-header>
            <mat-card-title>{{ news.title }}</mat-card-title>
            <mat-card-subtitle>
              <mat-chip>{{ getSubjectName(news.subject_id) }}</mat-chip>
              <span class="news-date">{{ news.created_at | russianDate:'datetime' }}</span>
            </mat-card-subtitle>
          </mat-card-header>
          <mat-card-content>
            <p>{{ news.content }}</p>
          </mat-card-content>
          <mat-card-actions>
            <button mat-button color="primary" (click)="editNews(news)">
              <mat-icon>edit</mat-icon>
              Редактировать
            </button>
            <button mat-button color="warn" (click)="deleteNews(news.id)">
              <mat-icon>delete</mat-icon>
              Удалить
            </button>
          </mat-card-actions>
        </mat-card>
      </div>
    </div>
  `,
  styles: [`
    .news-manage-container {
      max-width: 1000px;
      margin: 0 auto;
      padding: 20px;
    }
    
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 20px;
    }
    
    .news-list {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    
    .news-item {
      margin-bottom: 20px;
    }
    
    .news-image {
      width: 100%;
      height: 200px;
      object-fit: cover;
    }
    
    .news-date {
      margin-left: 10px;
      color: #666;
      font-size: 14px;
    }
    
    .empty-state {
      text-align: center;
      padding: 40px;
      color: #999;
    }
  `]
})
export class NewsManageComponent implements OnInit {
  newsForm: FormGroup;
  subjects: any[] = [];
  allNews: any[] = [];
  editingNews: any = null;
  submitting = false;

  constructor(
    private fb: FormBuilder,
    private apiService: ApiService
  ) {
    this.newsForm = this.fb.group({
      subject_id: ['', Validators.required],
      title: ['', Validators.required],
      content: ['', Validators.required],
      image_url: ['']
    });
  }

  ngOnInit() {
    this.loadSubjects();
    this.loadAllNews();
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  loadAllNews() {
    this.apiService.getNews().subscribe({
      next: (news) => {
        this.allNews = news;
      },
      error: (err) => {
        console.error('Error loading news:', err);
        this.allNews = [];
      }
    });
  }

  getSubjectName(subjectId: string): string {
    const subject = this.subjects.find(s => s.id === subjectId);
    return subject ? subject.name : 'Неизвестный курс';
  }

  onSubmit() {
    if (this.newsForm.valid) {
      this.submitting = true;
      const newsData = this.newsForm.value;

      if (this.editingNews) {
        // Обновление существующей новости
        this.apiService.updateNews(this.editingNews.id, newsData).subscribe({
          next: () => {
            this.submitting = false;
            this.resetForm();
            this.loadAllNews();
          },
          error: (err) => {
            console.error('Error updating news:', err);
            alert('Ошибка при обновлении новости');
            this.submitting = false;
          }
        });
      } else {
        // Создание новой новости
        this.apiService.createNews(newsData).subscribe({
          next: () => {
            this.submitting = false;
            this.newsForm.reset();
            this.loadAllNews();
          },
          error: (err) => {
            console.error('Error creating news:', err);
            alert('Ошибка при создании новости');
            this.submitting = false;
          }
        });
      }
    }
  }

  editNews(news: any) {
    this.editingNews = news;
    this.newsForm.patchValue({
      subject_id: news.subject_id,
      title: news.title,
      content: news.content,
      image_url: news.image_url || ''
    });
    // Прокручиваем к форме
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  deleteNews(newsId: string) {
    if (confirm('Вы уверены, что хотите удалить эту новость?')) {
      this.apiService.deleteNews(newsId).subscribe({
        next: () => {
          this.loadAllNews();
        },
        error: (err) => {
          console.error('Error deleting news:', err);
          alert('Ошибка при удалении новости');
        }
      });
    }
  }

  resetForm() {
    this.editingNews = null;
    this.newsForm.reset();
  }
}
