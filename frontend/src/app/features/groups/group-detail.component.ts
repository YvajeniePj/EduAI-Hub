import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTableModule } from '@angular/material/table';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';

@Component({
    selector: 'app-group-detail',
    standalone: true,
    imports: [
        CommonModule,
        RouterModule,
        MatCardModule,
        MatButtonModule,
        MatIconModule,
        MatTabsModule,
        MatTableModule,
        MatChipsModule,
        MatTooltipModule,
        MatProgressSpinnerModule
    ],
    template: `
    <div class="container" *ngIf="group">
      <div class="header">
        <button mat-icon-button routerLink="/groups">
          <mat-icon>arrow_back</mat-icon>
        </button>
        <div class="title-section">
          <h1>{{ group.name }}</h1>
          <p class="subtitle">{{ subjectName }}</p>
        </div>
      </div>

      <mat-tab-group class="content-tabs">
        <mat-tab label="Обзор">
          <div class="tab-content">
            <mat-card class="detail-card">
              <mat-card-content>
                <div class="info-grid">
                  <div class="info-item">
                    <span class="label">Описание</span>
                    <p>{{ group.description || 'Нет описания' }}</p>
                  </div>
                  <div class="info-item">
                    <span class="label">Макс. участников</span>
                    <p>{{ group.max_size || 'Нет ограничений' }}</p>
                  </div>
                  <div class="info-item">
                    <span class="label">Создатель</span>
                    <p>{{ group.created_by }}</p>
                  </div>
                </div>
              </mat-card-content>
            </mat-card>
          </div>
        </mat-tab>

        <mat-tab label="Участники">
          <ng-template matTabContent>
            <div class="tab-content">
              <div class="section-actions" *ngIf="isOwner">
                <button mat-raised-button color="primary">
                  <mat-icon>person_add</mat-icon> Добавить
                </button>
              </div>

              <div *ngIf="loadingMembers" class="loader">
                <mat-spinner diameter="40"></mat-spinner>
              </div>

              <div *ngIf="!loadingMembers && members.length === 0" class="empty-state">
                Нет участников в этой группе
              </div>

              <table mat-table [dataSource]="members" *ngIf="!loadingMembers && members.length > 0" class="members-table">
                <ng-container matColumnDef="name">
                  <th mat-header-cell *matHeaderCellDef>Имя</th>
                  <td mat-cell *matCellDef="let member">{{ member.user_name }}</td>
                </ng-container>
                <ng-container matColumnDef="joined">
                  <th mat-header-cell *matHeaderCellDef>Дата вступления</th>
                  <td mat-cell *matCellDef="let member">{{ member.joined_at | date:'dd.MM.yyyy' }}</td>
                </ng-container>
                <ng-container matColumnDef="actions">
                  <th mat-header-cell *matHeaderCellDef>Действия</th>
                  <td mat-cell *matCellDef="let member">
                    <button mat-icon-button color="warn" *ngIf="isOwner" (click)="removeMember(member.id)">
                      <mat-icon>delete</mat-icon>
                    </button>
                  </td>
                </ng-container>

                <tr mat-header-row *matHeaderRowDef="['name', 'joined', 'actions']"></tr>
                <tr mat-row *matRowDef="let row; columns: ['name', 'joined', 'actions'];"></tr>
              </table>
            </div>
          </ng-template>
        </mat-tab>

        <mat-tab label="Заявки" *ngIf="isOwner">
          <ng-template matTabContent>
            <div class="tab-content">
              <div *ngIf="loadingRequests" class="loader">
                <mat-spinner diameter="40"></mat-spinner>
              </div>

              <div *ngIf="!loadingRequests && requests.length === 0" class="empty-state">
                Нет ожидающих заявок
              </div>

              <div class="requests-grid" *ngIf="!loadingRequests && requests.length > 0">
                <mat-card *ngFor="let request of requests" class="request-card">
                  <mat-card-content>
                    <div class="request-info">
                      <div class="user-avatar">
                        <mat-icon>person</mat-icon>
                      </div>
                      <div class="user-details">
                        <span class="user-name">{{ request.user_name }}</span>
                        <span class="request-date">{{ request.created_at | date:'short' }}</span>
                      </div>
                    </div>
                    <div class="request-actions">
                      <button mat-raised-button color="primary" (click)="approve(request.id)">Принять</button>
                      <button mat-button color="warn" (click)="reject(request.id)">Отклонить</button>
                    </div>
                  </mat-card-content>
                </mat-card>
              </div>
            </div>
          </ng-template>
        </mat-tab>
      </mat-tab-group>
    </div>

    <div *ngIf="!group && !loading" class="empty-state">
      Группа не найдена
    </div>
  `,
    styles: [`
    .container {
      max-width: 1000px;
      margin: 24px auto;
      padding: 0 16px;
    }
    .header {
      display: flex;
      align-items: center;
      gap: 16px;
      margin-bottom: 24px;
    }
    .title-section h1 {
      margin: 0;
      font-size: 28px;
    }
    .subtitle {
      color: #666;
      margin: 4px 0 0 0;
    }
    .tab-content {
      padding: 24px 0;
    }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 24px;
    }
    .label {
      color: #999;
      font-size: 12px;
      text-transform: uppercase;
      display: block;
      margin-bottom: 4px;
    }
    .members-table {
      width: 100%;
      background: transparent;
    }
    .section-actions {
      margin-bottom: 16px;
      display: flex;
      justify-content: flex-end;
    }
    .loader {
      display: flex;
      justify-content: center;
      padding: 40px;
    }
    .empty-state {
      text-align: center;
      padding: 80px 20px;
      color: #999;
      background: white;
      border-radius: 8px;
    }
    .requests-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 16px;
    }
    .request-card {
      border-radius: 12px;
    }
    .request-card .mat-mdc-card-content {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .request-info {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .user-avatar {
      width: 40px;
      height: 40px;
      background: #f0f0f0;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #999;
    }
    .user-details {
      display: flex;
      flex-direction: column;
    }
    .user-name {
      font-weight: 500;
    }
    .request-date {
      font-size: 12px;
      color: #999;
    }
    .request-actions {
      display: flex;
      gap: 8px;
    }
  `]
})
export class GroupDetailComponent implements OnInit {
    groupId: string = '';
    group: any;
    subjectName: string = '';
    members: any[] = [];
    requests: any[] = [];
    loading = true;
    loadingMembers = false;
    loadingRequests = false;
    currentUser: any;
    isOwner = false;

    constructor(
        private route: ActivatedRoute,
        private api: ApiService,
        private auth: AuthService
    ) { }

    ngOnInit(): void {
        this.currentUser = this.auth.getCurrentUser();
        this.route.params.subscribe(params => {
            this.groupId = params['id'];
            this.loadGroup();
        });
    }

    loadGroup() {
        this.loading = true;
        this.api.getGroup(this.groupId).subscribe({
            next: (group) => {
                this.group = group;
                this.isOwner = group.created_by === this.currentUser?.name;
                this.loadSubject(group.subject_id);
                this.loadMembers();
                if (this.isOwner) {
                    this.loadRequests();
                }
                this.loading = false;
            },
            error: (err) => {
                console.error('Error loading group:', err);
                this.loading = false;
            }
        });
    }

    loadSubject(subjectId: string) {
        this.api.getSubjects().subscribe(subjects => {
            const s = subjects.find(sub => sub.id === subjectId);
            this.subjectName = s ? s.name : 'Неизвестный курс';
        });
    }

    loadMembers() {
        this.loadingMembers = true;
        this.api.getGroupMembers(this.groupId).subscribe({
            next: (members) => {
                this.members = members;
                this.loadingMembers = false;
            },
            error: () => this.loadingMembers = false
        });
    }

    loadRequests() {
        this.loadingRequests = true;
        this.api.getGroupRequests(this.groupId, 'pending').subscribe({
            next: (requests) => {
                this.requests = requests;
                this.loadingRequests = false;
            },
            error: () => this.loadingRequests = false
        });
    }

    approve(requestId: string) {
        this.api.updateGroupRequest(this.groupId, requestId, 'approved', this.currentUser.name).subscribe({
            next: () => {
                this.loadRequests();
                this.loadMembers();
            }
        });
    }

    reject(requestId: string) {
        this.api.updateGroupRequest(this.groupId, requestId, 'rejected', this.currentUser.name).subscribe({
            next: () => this.loadRequests()
        });
    }

    removeMember(memberId: string) {
        if (confirm('Удалить участника?')) {
            this.api.removeGroupMember(this.groupId, memberId).subscribe({
                next: () => this.loadMembers()
            });
        }
    }
}
