import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ApiService } from '../../core/services/api.service';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatInputModule,
    MatSelectModule,
    MatListModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  template: `
    <div class="chat-container">
      <h1>💬 Чат-ассистент</h1>
      
      <mat-card *ngIf="subjects.length > 0">
        <mat-card-content>
          <mat-form-field>
            <mat-label>Выберите предмет (опционально)</mat-label>
            <mat-select [(ngModel)]="selectedSubjectId">
              <mat-option [value]="null">Все предметы</mat-option>
              <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                {{ subject.name }}
              </mat-option>
            </mat-select>
          </mat-form-field>
        </mat-card-content>
      </mat-card>

      <mat-card class="chat-messages">
        <mat-card-content>
          <div class="messages-container">
            <div *ngIf="messages.length === 0" class="welcome-message">
              <p>Задайте вопрос ассистенту! Он поможет вам с:</p>
              <ul>
                <li>📅 Дедлайнами и сроками</li>
                <li>📚 Рекомендациями материалов</li>
                <li>🎥 Подходящими видео</li>
                <li>📝 Информацией о тестах</li>
                <li>💡 Общими вопросами по курсу</li>
              </ul>
            </div>
            
            <div *ngFor="let message of messages" [class]="'message ' + message.role">
              <div class="message-header">
                <strong>{{ message.role === 'user' ? 'Вы' : 'Ассистент' }}</strong>
                <span class="timestamp">{{ formatTime(message.timestamp) }}</span>
              </div>
              <div class="message-content" [innerHTML]="formatMessage(message.content)"></div>
            </div>
            
            <div *ngIf="loading" class="loading-indicator">
              <mat-spinner diameter="30"></mat-spinner>
              <span>Ассистент думает...</span>
            </div>
          </div>
        </mat-card-content>
      </mat-card>

      <mat-card class="chat-input">
        <mat-card-content>
          <div class="input-container">
            <mat-form-field style="flex: 1;">
              <mat-label>Ваш вопрос</mat-label>
              <input 
                matInput 
                [(ngModel)]="currentQuestion" 
                (keyup.enter)="sendMessage()"
                placeholder="Например: Какие материалы мне нужно изучить для теста?">
            </mat-form-field>
            <button 
              mat-raised-button 
              color="primary" 
              (click)="sendMessage()" 
              [disabled]="!currentQuestion.trim() || loading"
              style="margin-left: 10px;">
              <mat-icon>send</mat-icon>
              Отправить
            </button>
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [`
    .chat-container {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      height: calc(100vh - 100px);
    }
    mat-card {
      margin-bottom: 20px;
    }
    .chat-messages {
      flex: 1;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .messages-container {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
      max-height: 500px;
    }
    .message {
      margin-bottom: 20px;
      padding: 15px;
      border-radius: 8px;
    }
    .message.user {
      background-color: #e3f2fd;
      margin-left: 20%;
    }
    .message.assistant {
      background-color: #f5f5f5;
      margin-right: 20%;
    }
    .message-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 0.9em;
      color: #666;
    }
    .message-content {
      white-space: pre-wrap;
      line-height: 1.6;
    }
    .timestamp {
      font-size: 0.8em;
    }
    .welcome-message {
      text-align: center;
      padding: 40px;
      color: #666;
    }
    .welcome-message ul {
      text-align: left;
      display: inline-block;
    }
    .loading-indicator {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 20px;
      color: #666;
    }
    .chat-input {
      position: sticky;
      bottom: 0;
    }
    .input-container {
      display: flex;
      align-items: center;
    }
  `]
})
export class ChatComponent implements OnInit {
  subjects: any[] = [];
  messages: ChatMessage[] = [];
  currentQuestion: string = '';
  selectedSubjectId: string | null = null;
  loading: boolean = false;

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.loadSubjects();
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  sendMessage() {
    if (!this.currentQuestion.trim() || this.loading) return;

    const question = this.currentQuestion.trim();
    this.currentQuestion = '';

    // Add user message
    this.messages.push({
      role: 'user',
      content: question,
      timestamp: new Date()
    });

    // Scroll to bottom
    setTimeout(() => this.scrollToBottom(), 100);

    // Send to API
    this.loading = true;
    this.apiService.chat(question, this.selectedSubjectId || undefined).subscribe({
      next: (response) => {
        this.loading = false;
        this.messages.push({
          role: 'assistant',
          content: response.message || response.answer || 'Извините, не удалось получить ответ.',
          timestamp: new Date()
        });
        setTimeout(() => this.scrollToBottom(), 100);
      },
      error: (err) => {
        this.loading = false;
        console.error('Error sending message:', err);
        this.messages.push({
          role: 'assistant',
          content: 'Извините, произошла ошибка: ' + (err.error?.detail || err.message),
          timestamp: new Date()
        });
        setTimeout(() => this.scrollToBottom(), 100);
      }
    });
  }

  formatMessage(content: string): string {
    // Convert markdown-like formatting to HTML
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  }

  formatTime(date: Date): string {
    return date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
  }

  scrollToBottom() {
    const container = document.querySelector('.messages-container');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }
}

