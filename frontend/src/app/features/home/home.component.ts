import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { RussianDatePipe } from '../../core/pipes/russian-date.pipe';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatChipsModule,
    MatTooltipModule,
    RussianDatePipe
  ],
  template: `
    <div class="home-container">
      <h1>Добро пожаловать в EduAI Hub</h1>
      
      <!-- Новости -->
      <section class="news-section">
        <h2>Новости</h2>
        <div *ngIf="news.length === 0" class="empty-state">
          <p>Пока нет новостей</p>
        </div>
        <div class="news-grid" *ngIf="news.length > 0">
          <mat-card *ngFor="let item of news" class="news-card">
            <img *ngIf="item.image_url" [src]="item.image_url" alt="News image" class="news-image">
            <mat-card-header>
              <mat-card-title>{{ item.title }}</mat-card-title>
              <mat-card-subtitle>
                <mat-chip>{{ getSubjectName(item.subject_id) }}</mat-chip>
                <span class="news-date">{{ item.created_at | russianDate:'datetime' }}</span>
              </mat-card-subtitle>
            </mat-card-header>
            <mat-card-content>
              <p>{{ item.content }}</p>
            </mat-card-content>
          </mat-card>
        </div>
      </section>

      <!-- Календарь дедлайнов -->
      <section class="deadlines-section">
        <h2>Календарь дедлайнов</h2>
        <mat-card class="calendar-card">
          <mat-card-content>
            <div class="calendar-header">
              <button mat-icon-button (click)="previousMonth()">
                <mat-icon>chevron_left</mat-icon>
              </button>
              <h3 class="calendar-month">{{ getMonthYearLabel() }}</h3>
              <button mat-icon-button (click)="nextMonth()">
                <mat-icon>chevron_right</mat-icon>
              </button>
            </div>
            <div class="calendar-grid">
              <div class="calendar-weekday" *ngFor="let day of weekDays">{{ day }}</div>
              <div 
                *ngFor="let day of calendarDays" 
                class="calendar-day"
                [class.has-deadline]="hasDeadlineOnDate(day)"
                [class.overdue]="isDateOverdue(day)"
                [class.soon]="isDateSoon(day)"
                [class.other-month]="day.otherMonth"
                [matTooltip]="getDeadlinesTooltip(day)"
                [matTooltipPosition]="'above'"
                [matTooltipShowDelay]="200"
                [matTooltipClass]="'deadline-tooltip'">
                <span class="day-number">{{ day.date }}</span>
                <div *ngIf="hasDeadlineOnDate(day)" class="deadline-indicator"></div>
                <div *ngIf="hasDeadlineOnDate(day)" class="deadline-count">{{ getDeadlineCount(day) }}</div>
              </div>
            </div>
          </mat-card-content>
        </mat-card>
      </section>
    </div>
  `,
  styles: [`
    .home-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    
    h1 {
      font-size: 32px;
      margin-bottom: 30px;
    }
    
    h2 {
      font-size: 24px;
      margin: 40px 0 20px 0;
    }
    
    .news-section, .deadlines-section {
      margin-bottom: 40px;
    }
    
    .news-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 20px;
    }
    
    .news-card {
      display: flex;
      flex-direction: column;
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
    
    .calendar-card {
      max-width: 800px;
      margin: 0 auto;
    }
    
    .calendar-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
    }
    
    .calendar-month {
      margin: 0;
      font-size: 20px;
      font-weight: 500;
    }
    
    .calendar-grid {
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 4px;
    }
    
    .calendar-weekday {
      text-align: center;
      font-weight: 600;
      padding: 8px;
      color: #666;
      font-size: 14px;
    }
    
    .calendar-day {
      aspect-ratio: 1;
      border: 1px solid #e0e0e0;
      border-radius: 4px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      position: relative;
      cursor: pointer;
      transition: all 0.2s;
      background: white;
    }
    
    .calendar-day:hover {
      background: #f5f5f5;
      border-color: #3f51b5;
    }
    
    .calendar-day.other-month {
      opacity: 0.3;
      background: #fafafa;
    }
    
    .calendar-day.has-deadline {
      background: #e3f2fd;
      border-color: #2196f3;
      font-weight: 600;
    }
    
    .calendar-day.has-deadline.overdue {
      background: #ffebee;
      border-color: #f44336;
      color: #f44336;
    }
    
    .calendar-day.has-deadline.soon {
      background: #fff3e0;
      border-color: #ff9800;
      color: #ff9800;
    }
    
    .day-number {
      font-size: 16px;
    }
    
    .deadline-indicator {
      position: absolute;
      bottom: 4px;
      width: 6px;
      height: 6px;
      background: #3f51b5;
      border-radius: 50%;
    }
    
    .calendar-day.overdue .deadline-indicator {
      background: #f44336;
    }
    
    .calendar-day.soon .deadline-indicator {
      background: #ff9800;
    }
    
    .deadline-count {
      position: absolute;
      top: 2px;
      right: 2px;
      background: #3f51b5;
      color: white;
      border-radius: 50%;
      width: 18px;
      height: 18px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-weight: bold;
    }
    
    .calendar-day.overdue .deadline-count {
      background: #f44336;
    }
    
    .calendar-day.soon .deadline-count {
      background: #ff9800;
    }
    
    ::ng-deep .deadline-tooltip {
      white-space: pre-line;
      max-width: 300px;
      font-size: 13px;
      line-height: 1.5;
    }
    
    .empty-state {
      text-align: center;
      padding: 40px;
      color: #999;
    }
  `]
})
export class HomeComponent implements OnInit {
  news: any[] = [];
  deadlines: any[] = [];
  subjects: any[] = [];
  currentDate: Date = new Date();
  calendarDays: any[] = [];
  weekDays = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];

  constructor(
    private apiService: ApiService,
    private authService: AuthService
  ) {}

  ngOnInit() {
    // Сначала загружаем предметы, затем новости
    // Дедлайны загрузятся после предметов в loadSubjects
    this.loadSubjects();
    this.loadNews();
  }

  loadNews() {
    this.apiService.getNews().subscribe({
      next: (news) => {
        this.news = news;
      },
      error: (err) => {
        console.error('Error loading news:', err);
        this.news = [];
      }
    });
  }

  loadDeadlines() {
    this.apiService.getTests().subscribe({
      next: (tests) => {
        // Фильтруем тесты с дедлайнами
        this.deadlines = tests
          .filter(t => t.due_date)
          .map(t => ({
            title: t.title,
            due_date: t.due_date,
            subject_id: t.subject_id,
            test_id: t.id,
            subject_name: this.getSubjectName(t.subject_id)
          }))
          .sort((a, b) => new Date(a.due_date).getTime() - new Date(b.due_date).getTime());
        this.generateCalendar();
      },
      error: (err) => {
        console.error('Error loading deadlines:', err);
        this.deadlines = [];
        this.generateCalendar();
      }
    });
  }

  generateCalendar() {
    const year = this.currentDate.getFullYear();
    const month = this.currentDate.getMonth();
    
    // Первый день месяца
    const firstDay = new Date(year, month, 1);
    // Последний день месяца
    const lastDay = new Date(year, month + 1, 0);
    
    // День недели первого дня (0 = воскресенье, нужно преобразовать: 0 -> 6, 1-6 -> 0-5)
    let startDay = firstDay.getDay();
    startDay = startDay === 0 ? 6 : startDay - 1; // Понедельник = 0
    
    // День недели последнего дня
    let endDay = lastDay.getDay();
    endDay = endDay === 0 ? 6 : endDay - 1;
    
    const days: any[] = [];
    
    // Дни предыдущего месяца
    const prevMonthLastDay = new Date(year, month, 0).getDate();
    for (let i = startDay - 1; i >= 0; i--) {
      days.push({
        date: prevMonthLastDay - i,
        fullDate: new Date(year, month - 1, prevMonthLastDay - i),
        otherMonth: true
      });
    }
    
    // Дни текущего месяца
    for (let i = 1; i <= lastDay.getDate(); i++) {
      days.push({
        date: i,
        fullDate: new Date(year, month, i),
        otherMonth: false
      });
    }
    
    // Дни следующего месяца
    const daysToAdd = 42 - days.length; // 6 недель * 7 дней = 42
    for (let i = 1; i <= daysToAdd; i++) {
      days.push({
        date: i,
        fullDate: new Date(year, month + 1, i),
        otherMonth: true
      });
    }
    
    this.calendarDays = days;
  }

  previousMonth() {
    this.currentDate = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth() - 1, 1);
    this.generateCalendar();
  }

  nextMonth() {
    this.currentDate = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth() + 1, 1);
    this.generateCalendar();
  }

  getMonthYearLabel(): string {
    const months = [
      'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
      'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'
    ];
    return `${months[this.currentDate.getMonth()]} ${this.currentDate.getFullYear()}`;
  }

  hasDeadlineOnDate(day: any): boolean {
    if (day.otherMonth) return false;
    const dayStr = this.formatDateForComparison(day.fullDate);
    return this.deadlines.some(d => {
      const deadlineDate = new Date(d.due_date);
      return this.formatDateForComparison(deadlineDate) === dayStr;
    });
  }

  isDateOverdue(day: any): boolean {
    if (day.otherMonth) return false;
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const dayDate = new Date(day.fullDate);
    dayDate.setHours(0, 0, 0, 0);
    return dayDate < today && this.hasDeadlineOnDate(day);
  }

  isDateSoon(day: any): boolean {
    if (day.otherMonth || this.isDateOverdue(day)) return false;
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const dayDate = new Date(day.fullDate);
    dayDate.setHours(0, 0, 0, 0);
    const daysDiff = Math.ceil((dayDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
    return daysDiff <= 7 && daysDiff > 0 && this.hasDeadlineOnDate(day);
  }

  getDeadlinesTooltip(day: any): string {
    if (!this.hasDeadlineOnDate(day)) return '';
    const dayStr = this.formatDateForComparison(day.fullDate);
    const dayDeadlines = this.deadlines.filter(d => {
      const deadlineDate = new Date(d.due_date);
      return this.formatDateForComparison(deadlineDate) === dayStr;
    });
    
    if (dayDeadlines.length === 0) return '';
    
    return dayDeadlines.map(d => {
      const deadlineDate = new Date(d.due_date);
      const now = new Date();
      
      // Дата уже в московском времени, просто сравниваем
      const isExpired = deadlineDate.getTime() < now.getTime();
      
      const status = isExpired ? '❌ Закрыт' : '✅ Открыт';
      const dateStr = this.formatRussianDate(deadlineDate);
      const timeStr = this.formatRussianTime(deadlineDate);
      
      return `${d.title}\n${status} | До ${timeStr} (${dateStr})\n${d.subject_name || 'Без предмета'}`;
    }).join('\n\n');
  }

  formatDateForComparison(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  formatRussianDate(date: Date): string {
    // Дата уже в московском времени, просто форматируем
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    return `${day}.${month}.${year}`;
  }

  formatRussianTime(date: Date): string {
    // Дата уже в московском времени, просто форматируем
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
  }

  getDeadlineCount(day: any): number {
    if (!this.hasDeadlineOnDate(day)) return 0;
    const dayStr = this.formatDateForComparison(day.fullDate);
    return this.deadlines.filter(d => {
      const deadlineDate = new Date(d.due_date);
      return this.formatDateForComparison(deadlineDate) === dayStr;
    }).length;
  }


  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
        // После загрузки предметов загружаем дедлайны, чтобы правильно отобразить названия предметов
        this.loadDeadlines();
      },
      error: (err) => {
        console.error('Error loading subjects:', err);
        this.loadDeadlines(); // Загружаем дедлайны даже если предметы не загрузились
      }
    });
  }

  getSubjectName(subjectId: string): string {
    const subject = this.subjects.find(s => s.id === subjectId);
    return subject ? subject.name : 'Неизвестный предмет';
  }

  isOverdue(date: string): boolean {
    return new Date(date) < new Date();
  }

  isSoon(date: string): boolean {
    const deadline = new Date(date);
    const now = new Date();
    const daysDiff = Math.ceil((deadline.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
    return daysDiff <= 7 && daysDiff > 0;
  }
}
