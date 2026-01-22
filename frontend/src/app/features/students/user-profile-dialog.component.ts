import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatDialogModule, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';

@Component({
    selector: 'app-user-profile-dialog',
    standalone: true,
    imports: [CommonModule, MatDialogModule, MatButtonModule, MatIconModule, MatChipsModule],
    template: `
    <div class="profile-dialog-header">
      <div class="avatar-container">
        <img [src]="data.user.avatar_url || 'assets/default-avatar.png'" alt="avatar" class="avatar">
      </div>
      <h2>{{ data.user.name }}</h2>
      <span class="role-badge" [class.teacher]="data.user.role === 'teacher'">
        {{ data.user.role === 'teacher' ? 'Преподаватель' : 'Студент' }}
      </span>
    </div>
    
    <mat-dialog-content>
      <div class="info-section">
        <h3>Группы</h3>
        <div class="groups-list" *ngIf="data.user.groups && data.user.groups.length > 0; else noGroups">
          <mat-chip-set>
            <mat-chip *ngFor="let group of data.user.groups">
              {{ group.name }}
            </mat-chip>
          </mat-chip-set>
        </div>
        <ng-template #noGroups>
          <p class="empty-text">Пользователь не состоит в группах</p>
        </ng-template>
      </div>
    </mat-dialog-content>

    <mat-dialog-actions align="end">
      <button mat-button mat-dialog-close>Закрыть</button>
    </mat-dialog-actions>
  `,
    styles: [`
    .profile-dialog-header {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 24px 0;
    }
    .avatar-container {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      overflow: hidden;
      margin-bottom: 16px;
      background-color: #f0f0f0;
      border: 3px solid white;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .avatar {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    h2 {
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 500;
      color: #333;
    }
    .role-badge {
      font-size: 12px;
      padding: 4px 12px;
      border-radius: 16px;
      background: #eee;
      color: #666;
      text-transform: uppercase;
      font-weight: bold;
      letter-spacing: 0.5px;
    }
    .role-badge.teacher {
      background: #e8f5e9;
      color: #2e7d32;
    }
    .info-section {
      margin-top: 24px;
      min-width: 300px;
    }
    h3 {
      font-size: 14px;
      font-weight: 500;
      margin-bottom: 12px;
      color: #666;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .empty-text {
      color: #999;
      font-style: italic;
      font-size: 14px;
    }
  `]
})
export class UserProfileDialogComponent {
    constructor(@Inject(MAT_DIALOG_DATA) public data: { user: any }) { }
}
