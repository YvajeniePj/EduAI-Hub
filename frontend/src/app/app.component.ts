import { Component, OnInit, OnDestroy, HostListener } from '@angular/core';
import { RouterOutlet, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatButtonModule } from '@angular/material/button';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatBadgeModule } from '@angular/material/badge';
import { MatMenuModule } from '@angular/material/menu';
import { ApiService } from './core/services/api.service';
import { AuthService, CurrentUser } from './core/services/auth.service';
import { Router } from '@angular/router';
import { interval, Subscription } from 'rxjs';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterOutlet,
    RouterModule,
    MatToolbarModule,
    MatButtonModule,
    MatSidenavModule,
    MatListModule,
    MatIconModule,
    MatTooltipModule,
    MatBadgeModule,
    MatMenuModule
  ],
  template: `
    <mat-sidenav-container class="sidenav-container" *ngIf="currentUser; else noAuth">
      <mat-sidenav #sidenav mode="over" class="app-sidenav">
        <div class="sidenav-header">
          <mat-icon class="sidenav-logo-icon">school</mat-icon>
          <span class="sidenav-logo-text">EduAI Hub</span>
          <button mat-icon-button (click)="sidenav.close()">
            <mat-icon>close</mat-icon>
          </button>
        </div>
        
        <mat-nav-list>
          <!-- Home -->
          <a mat-list-item routerLink="/" (click)="sidenav.close()" routerLinkActive="active-link" [routerLinkActiveOptions]="{exact: true}">
            <mat-icon matListItemIcon>home</mat-icon>
            <span matListItemTitle>Главная</span>
          </a>

          <div class="nav-divider"></div>
          
          <!-- Block: Tests -->
          <div class="nav-block-header">ТЕСТЫ</div>
          <a mat-list-item routerLink="/tests" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>quiz</mat-icon>
            <span matListItemTitle>Тесты</span>
          </a>
          <a mat-list-item routerLink="/ai-test" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>psychology</mat-icon>
            <span matListItemTitle>AI-генерация</span>
          </a>
          <a mat-list-item routerLink="/peer-review" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>rate_review</mat-icon>
            <span matListItemTitle>Кросс-проверка</span>
          </a>

          <div class="nav-divider"></div>

          <!-- Block: Courses -->
          <div class="nav-block-header">КУРСЫ</div>
          <a mat-list-item routerLink="/subjects" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>library_books</mat-icon>
            <span matListItemTitle>Курсы</span>
          </a>
          <a mat-list-item routerLink="/tests/create" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>add_task</mat-icon>
            <span matListItemTitle>Конструктор тестов</span>
          </a>
          <a mat-list-item routerLink="/course-builder" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>construction</mat-icon>
            <span matListItemTitle>Конструктор курсов</span>
          </a>

          <div class="nav-divider"></div>

          <!-- Block: Materials -->
          <div class="nav-block-header">МАТЕРИАЛЫ</div>
          <a mat-list-item routerLink="/materials" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>folder</mat-icon>
            <span matListItemTitle>Материалы</span>
          </a>
          <a mat-list-item routerLink="/videos" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>video_library</mat-icon>
            <span matListItemTitle>Видео</span>
          </a>

          <div class="nav-divider"></div>

          <!-- Block: Analytics (Teachers Only) -->
          <ng-container *ngIf="currentUser.role === 'teacher'">
          <div class="nav-block-header">АНАЛИТИКА</div>
          <a mat-list-item routerLink="/analytics" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>insights</mat-icon>
            <span matListItemTitle>Аналитика</span>
          </a>
          <a mat-list-item routerLink="/activity-monitor" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>visibility</mat-icon>
            <span matListItemTitle>Мониторинг</span>
          </a>
          <a mat-list-item routerLink="/feedback-results" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>reviews</mat-icon>
            <span matListItemTitle>Отзывы</span>
          </a>
          <div class="nav-divider"></div>
          </ng-container>

          <!-- Block: Students -->
          <div class="nav-block-header">СТУДЕНТАМ</div>
          <a mat-list-item routerLink="/groups" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>groups</mat-icon>
            <span matListItemTitle>Группы</span>
          </a>
          <a mat-list-item routerLink="/students" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>person_search</mat-icon>
            <span matListItemTitle>Пользователи</span>
          </a>
          <a mat-list-item routerLink="/news/manage" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>feed</mat-icon>
            <span matListItemTitle>Новости</span>
          </a>
          <a mat-list-item routerLink="/leaderboard" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>emoji_events</mat-icon>
            <span matListItemTitle>Лидерборд</span>
          </a>
          <a mat-list-item routerLink="/chat" (click)="sidenav.close()" routerLinkActive="active-link">
            <mat-icon matListItemIcon>smart_toy</mat-icon>
            <span matListItemTitle>Чат-ассистент</span>
          </a>

          <div class="nav-divider"></div>
          
          <a mat-list-item routerLink="/admin/db" (click)="sidenav.close()" routerLinkActive="active-link" *ngIf="currentUser.role === 'teacher'">
            <mat-icon matListItemIcon>database</mat-icon>
            <span matListItemTitle>База данных</span>
          </a>
        </mat-nav-list>
      </mat-sidenav>

      <mat-sidenav-content>
        <mat-toolbar color="primary" class="app-toolbar">
          <button mat-icon-button (click)="sidenav.toggle()">
            <mat-icon>menu</mat-icon>
          </button>
          
          <mat-icon 
            [class.status-connected]="aiStatus?.available"
            [class.status-disconnected]="!aiStatus?.available"
            [matTooltip]="aiStatus?.message || 'Проверка статуса...'"
            style="margin-left: 16px; margin-right: 8px;">
            {{ aiStatus?.available ? 'check_circle' : 'error' }}
          </mat-icon>
          
          <span class="app-title">EduAI Hub</span>
          
          <span class="spacer"></span>
          
          <!-- Notifications -->
          <button mat-icon-button [matMenuTriggerFor]="notificationsMenu" class="notification-button">
            <mat-icon [matBadge]="unreadCount" [matBadgeHidden]="unreadCount === 0" matBadgeColor="warn">notifications</mat-icon>
          </button>
          
          <mat-menu #notificationsMenu="matMenu" class="notifications-menu">
            <div class="notifications-header">
              <h3>Уведомления</h3>
              <button mat-button *ngIf="unreadCount > 0" (click)="markAllRead()" class="mark-all-read">Отметить все прочитанными</button>
            </div>
            <div class="notifications-list">
              <div *ngIf="notifications.length === 0" class="no-notifications">
                Нет уведомлений
              </div>
              <div *ngFor="let notification of notifications" 
                   class="notification-item" 
                   [class.unread]="!notification.is_read"
                   (click)="markAsRead(notification.id)">
                <div class="notification-content">
                  <div class="notification-title">{{ notification.title || notification.type || 'Уведомление' }}</div>
                  <div class="notification-message">{{ notification.message }}</div>
                  <div class="notification-time">{{ formatNotificationTime(notification.created_at) }}</div>
                </div>
                <button mat-icon-button (click)="deleteNotification(notification.id, $event)" class="delete-notification">
                  <mat-icon>close</mat-icon>
                </button>
              </div>
            </div>
          </mat-menu>
          
          <button mat-button routerLink="/profile" class="profile-button">
            <div class="avatar-container" *ngIf="currentUser.avatar_url">
              <img [src]="currentUser.avatar_url" alt="avatar" class="toolbar-avatar">
            </div>
            <mat-icon *ngIf="!currentUser.avatar_url">person</mat-icon>
            {{ currentUser.name }}
          </button>
          <button mat-icon-button (click)="logout()" matTooltip="Выйти">
            <mat-icon>logout</mat-icon>
          </button>
        </mat-toolbar>

        <div class="container main-content">
          <router-outlet></router-outlet>
        </div>
      </mat-sidenav-content>
    </mat-sidenav-container>
    
    <ng-template #noAuth>
      <div class="container">
        <router-outlet></router-outlet>
      </div>
    </ng-template>
  `,
  styles: [`
    .spacer {
      flex: 1 1 auto;
    }
    .sidenav-container {
      height: 100vh;
      background: #f8f9fa;
    }
    .app-sidenav {
      width: 320px;
      box-shadow: 4px 0 10px rgba(0,0,0,0.05);
      border-right: none;
      background: white;
    }
    .sidenav-header {
      display: flex;
      align-items: center;
      padding: 20px 16px;
      background: #3f51b5;
      color: white;
    }
    .sidenav-logo-icon {
      font-size: 32px;
      width: 32px;
      height: 32px;
      margin-right: 12px;
    }
    .sidenav-logo-text {
      font-size: 20px;
      font-weight: 500;
      flex-grow: 1;
    }
    .nav-divider {
      height: 1px;
      background: #eee;
      margin: 8px 0;
    }
    .nav-block-header {
      padding: 16px 16px 8px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 1px;
      color: #999;
      text-transform: uppercase;
    }
    .active-link {
      background: rgba(63, 81, 181, 0.08);
      color: #3f51b5;
      border-left: 4px solid #3f51b5;
    }
    .app-toolbar {
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      z-index: 1000;
      position: sticky;
      top: 0;
    }
    .app-title {
      font-weight: 500;
      letter-spacing: 0.5px;
    }
    .main-content {
      padding: 24px;
      min-height: calc(100vh - 64px);
    }
    .container {
      max-width: 1400px;
      margin: 0 auto;
    }
    .avatar-container {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      overflow: hidden;
      margin-right: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #e0e0e0;
    }
    .toolbar-avatar {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .profile-button {
      display: flex;
      align-items: center;
      padding: 0 12px;
    }
    mat-icon.status-connected {
      color: #4caf50;
    }
    mat-icon.status-disconnected {
      color: #f44336;
    }
    .notification-button {
      margin-right: 8px;
    }
    ::ng-deep .notifications-menu {
      max-width: 400px;
      min-width: 350px;
      margin-top: 10px;
      border-radius: 8px !important;
    }
    .notifications-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid #f0f0f0;
      background: #fff;
    }
    .notifications-header h3 {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      color: #333;
    }
    .mark-all-read {
      font-size: 11px;
      line-height: normal;
      color: #3f51b5;
    }
    .notifications-list {
      max-height: 400px;
      overflow-y: auto;
    }
    .no-notifications {
      padding: 32px 16px;
      text-align: center;
      color: #999;
      font-size: 14px;
    }
    .notification-item {
      display: flex;
      align-items: flex-start;
      padding: 12px 16px;
      border-bottom: 1px solid #f9f9f9;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }
    .notification-item:hover {
      background-color: #fafafa;
    }
    .notification-item.unread {
      background-color: #f0f7ff;
      border-left: 3px solid #3f51b5;
    }
    .notification-item.unread:hover {
      background-color: #e8f0fe;
    }
    .notification-content {
      flex: 1;
      min-width: 0;
      padding-right: 8px;
    }
    .notification-title {
      font-weight: 600;
      margin-bottom: 4px;
      color: #2c3e50;
      font-size: 14px;
      line-height: 1.3;
    }
    .notification-message {
      font-size: 13px;
      color: #555;
      margin-bottom: 6px;
      line-height: 1.4;
      word-wrap: break-word;
    }
    .notification-time {
      font-size: 11px;
      color: #999;
      display: flex;
      align-items: center;
    }
    .delete-notification {
      width: 28px;
      height: 28px;
      line-height: 28px;
      opacity: 0.2;
      transition: opacity 0.2s;
    }
    .notification-item:hover .delete-notification {
      opacity: 0.6;
    }
    .delete-notification:hover {
      opacity: 1;
      color: #f44336;
    }
  `]
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'EduAI Hub';
  aiStatus: any = null;
  private statusCheckSubscription?: Subscription;
  private authSubscription?: Subscription;
  currentUser: CurrentUser | null = null;
  notifications: any[] = [];
  unreadCount: number = 0;
  private notificationCheckInterval?: Subscription;
  private sessionStartTime: number = Date.now();

  constructor(
    private apiService: ApiService,
    private auth: AuthService,
    private router: Router
  ) { }

  ngOnInit() {
    this.currentUser = this.auth.getCurrentUser();

    // Подписываемся на изменения текущего пользователя
    this.authSubscription = this.auth.currentUser$.subscribe(user => {
      const previouslyLoggedIn = !!this.currentUser;
      this.currentUser = user;

      if (user && !previouslyLoggedIn) {
        this.startSession();
      } else if (!user && previouslyLoggedIn) {
        // Session handled by logout() or beforeunload
      }
    });

    if (this.currentUser) {
      this.startSession();
    }

    this.checkAiStatus();
    // Check status every 30 seconds
    this.statusCheckSubscription = interval(30000).subscribe(() => {
      this.checkAiStatus();
    });

    // Load notifications
    this.loadNotifications();
    // Check notifications every 10 seconds
    this.notificationCheckInterval = interval(10000).subscribe(() => {
      this.loadNotifications();
    });
  }

  ngOnDestroy() {
    if (this.statusCheckSubscription) {
      this.statusCheckSubscription.unsubscribe();
    }
    if (this.authSubscription) {
      this.authSubscription.unsubscribe();
    }
    if (this.notificationCheckInterval) {
      this.notificationCheckInterval.unsubscribe();
    }
    this.endSession();
  }

  @HostListener('window:beforeunload', ['$event'])
  beforeUnloadHandler(event: any) {
    this.endSession();
  }

  private startSession() {
    if (!this.currentUser) return;
    this.sessionStartTime = Date.now();

    this.apiService.createActivity({
      user_name: this.currentUser.name,
      action_type: 'session_start'
    }).subscribe({
      error: (err) => console.error('Error tracking session start:', err)
    });
  }

  private endSession() {
    if (!this.currentUser) return;
    const duration = Math.floor((Date.now() - this.sessionStartTime) / 1000);

    // We use a beacon or a synchronous-like call if possible, but for now just try standard call
    // In a real app, we might use navigator.sendBeacon for beforeunload
    this.apiService.createActivity({
      user_name: this.currentUser.name,
      action_type: 'session_end',
      session_duration: duration
    }).subscribe();
  }

  loadNotifications() {
    if (!this.currentUser) return;

    this.apiService.getNotifications(this.currentUser.name, false).subscribe({
      next: (notifications) => {
        this.notifications = notifications.slice(0, 10); // Show latest 10
        this.unreadCount = notifications.filter((n: any) => !n.is_read).length;
      },
      error: (err) => {
        console.error('Error loading notifications:', err);
      }
    });
  }

  markAsRead(notificationId: string) {
    this.apiService.markNotificationRead(notificationId).subscribe({
      next: () => {
        const notification = this.notifications.find(n => n.id === notificationId);
        if (notification) {
          notification.is_read = true;
        }
        this.unreadCount = Math.max(0, this.unreadCount - 1);
      },
      error: (err) => {
        console.error('Error marking notification as read:', err);
      }
    });
  }

  markAllRead() {
    if (!this.currentUser) return;

    this.apiService.markAllNotificationsRead(this.currentUser.name).subscribe({
      next: () => {
        this.notifications.forEach(n => n.is_read = true);
        this.unreadCount = 0;
      },
      error: (err) => {
        console.error('Error marking all notifications as read:', err);
      }
    });
  }

  deleteNotification(notificationId: string, event: Event) {
    event.stopPropagation();
    this.apiService.deleteNotification(notificationId).subscribe({
      next: () => {
        const index = this.notifications.findIndex(n => n.id === notificationId);
        if (index !== -1) {
          if (!this.notifications[index].is_read) {
            this.unreadCount = Math.max(0, this.unreadCount - 1);
          }
          this.notifications.splice(index, 1);
        }
      },
      error: (err) => {
        console.error('Error deleting notification:', err);
      }
    });
  }

  formatNotificationTime(dateString: string): string {
    try {
      // Backend returns UTC but might be missing 'Z' or explicit offset in some cases
      // or browser interprets naive string as local.
      // Easiest fix: if it doesn't end in Z, append it to treat as UTC.
      let safeDateString = dateString;
      if (!dateString.endsWith('Z') && !dateString.includes('+')) {
        safeDateString = dateString + 'Z';
      }

      const date = new Date(safeDateString);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMs / 3600000);
      const diffDays = Math.floor(diffMs / 86400000);

      if (diffMins < 1) return 'только что';
      if (diffMins < 60) return `${diffMins} мин. назад`;
      if (diffHours < 24) return `${diffHours} ч. назад`;
      if (diffDays < 7) return `${diffDays} дн. назад`;

      return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' });
    } catch {
      return '';
    }
  }

  checkAiStatus() {
    this.apiService.getAiStatus().subscribe({
      next: (status) => {
        this.aiStatus = status;
      },
      error: (err) => {
        console.error('Error checking AI status:', err);
        this.aiStatus = {
          available: false,
          message: 'Ошибка проверки статуса'
        };
      }
    });
  }

  logout() {
    this.endSession();
    this.auth.logout();
    this.currentUser = null;
    this.router.navigate(['/login']);
  }
}

