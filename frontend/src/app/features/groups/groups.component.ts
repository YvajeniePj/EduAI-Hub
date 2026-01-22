import { Component, OnInit, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { MatTableModule } from '@angular/material/table';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatDialogModule, MatDialog, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatTabsModule } from '@angular/material/tabs';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-groups',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatButtonModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatTableModule,
    MatProgressSpinnerModule,
    MatDialogModule,
    MatTabsModule,
    MatChipsModule,
    MatTooltipModule
  ],
  template: `
    <div class="groups-container">
      <div class="groups-content">
        <div class="page-header">
          <h1 class="page-title">Управление группами</h1>
          <p class="page-subtitle">Создавайте группы для курсов и управляйте студентами</p>
        </div>

        <mat-tab-group>
          <!-- Tab 1: Мои группы -->
          <mat-tab label="Мои группы">
            <div class="tab-content">
              <mat-card class="filter-card">
                <mat-card-content>
                  <mat-form-field appearance="outline">
                    <mat-label>Курс</mat-label>
                    <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="loadMyGroups()">
                      <mat-option [value]="null">Все курсы</mat-option>
                      <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                        {{ subject.name }}
                      </mat-option>
                    </mat-select>
                  </mat-form-field>

                  <button mat-raised-button color="primary" (click)="openCreateGroupDialog()" class="create-button">
                    <mat-icon>add</mat-icon>
                    Создать группу
                  </button>
                </mat-card-content>
              </mat-card>

              <div *ngIf="loading" class="loading">
                <mat-spinner></mat-spinner>
              </div>

              <div *ngIf="!loading && myGroups.length === 0" class="empty-state">
                <mat-icon>group</mat-icon>
                <p>У вас нет созданных групп</p>
                <p class="empty-hint">Создайте первую группу для курса</p>
              </div>

              <div class="groups-grid" *ngIf="!loading && myGroups.length > 0">
                <mat-card *ngFor="let group of myGroups" class="group-card">
                  <mat-card-header>
                    <div class="group-header">
                      <div>
                        <mat-card-title>{{ group.name }}</mat-card-title>
                        <mat-card-subtitle>{{ getSubjectName(group.subject_id) }}</mat-card-subtitle>
                      </div>
                      <div class="group-actions">
                        <button mat-icon-button (click)="navigateToGroup(group.id)" matTooltip="Просмотр">
                          <mat-icon>visibility</mat-icon>
                        </button>
                        <button mat-icon-button color="warn" (click)="deleteGroup(group.id)" matTooltip="Удалить">
                          <mat-icon>delete</mat-icon>
                        </button>
                      </div>
                    </div>
                  </mat-card-header>
                  <mat-card-content>
                    <p *ngIf="group.description" class="group-description">{{ group.description }}</p>
                    <div class="group-stats">
                      <div class="stat-item">
                        <mat-icon>people</mat-icon>
                        <span>{{ group.member_count || 0 }} / {{ group.max_size || '∞' }}</span>
                      </div>
                      <div class="stat-item" *ngIf="group.created_by">
                        <mat-icon>person</mat-icon>
                        <span>Создатель: {{ group.created_by }}</span>
                      </div>
                    </div>
                  </mat-card-content>
                </mat-card>
              </div>
            </div>
          </mat-tab>

          <!-- Tab 2: Все группы -->
          <mat-tab label="Все группы">
            <div class="tab-content">
              <mat-card class="filter-card">
                <mat-card-content>
                  <mat-form-field appearance="outline">
                    <mat-label>Курс</mat-label>
                    <mat-select [(ngModel)]="selectedSubjectIdForAll" (selectionChange)="loadAllGroups()">
                      <mat-option [value]="null">Все курсы</mat-option>
                      <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                        {{ subject.name }}
                      </mat-option>
                    </mat-select>
                  </mat-form-field>
                </mat-card-content>
              </mat-card>

              <div *ngIf="loading" class="loading">
                <mat-spinner></mat-spinner>
              </div>

              <div *ngIf="!loading && allGroups.length === 0" class="empty-state">
                <mat-icon>group</mat-icon>
                <p>Нет доступных групп</p>
              </div>

              <div class="groups-grid" *ngIf="!loading && allGroups.length > 0">
                <mat-card *ngFor="let group of allGroups" class="group-card">
                  <mat-card-header>
                    <div class="group-header">
                      <div>
                        <mat-card-title>{{ group.name }}</mat-card-title>
                        <mat-card-subtitle>{{ getSubjectName(group.subject_id) }}</mat-card-subtitle>
                      </div>
                    </div>
                  </mat-card-header>
                  <mat-card-content>
                    <p *ngIf="group.description" class="group-description">{{ group.description }}</p>
                    <div class="group-stats">
                      <div class="stat-item">
                        <mat-icon>people</mat-icon>
                        <span>{{ group.member_count || 0 }} / {{ group.max_size || '∞' }}</span>
                      </div>
                    </div>
                    <div class="group-actions-bottom">
                      <button 
                        mat-raised-button 
                        color="primary" 
                        (click)="requestToJoinGroup(group)"
                        [disabled]="isRequestPending(group.id) || isGroupMember(group.id)"
                        matTooltip="Отправить заявку на вступление">
                        <mat-icon>person_add</mat-icon>
                        {{ isRequestPending(group.id) ? 'Заявка отправлена' : isGroupMember(group.id) ? 'Вы в группе' : 'Подать заявку' }}
                      </button>
                    </div>
                  </mat-card-content>
                </mat-card>
              </div>
            </div>
          </mat-tab>

          <!-- Tab 3: Мои заявки (Student) -->
          <mat-tab label="Мои заявки" *ngIf="currentUser?.role === 'student'">
            <div class="tab-content">
              <div *ngIf="loading" class="loading">
                <mat-spinner></mat-spinner>
              </div>

              <div *ngIf="!loading && myRequests.length === 0" class="empty-state">
                <mat-icon>inbox</mat-icon>
                <p>У вас нет заявок на вступление</p>
              </div>

              <div class="requests-list" *ngIf="!loading && myRequests.length > 0">
                <mat-card *ngFor="let request of myRequests" class="request-card">
                  <mat-card-content>
                    <div class="request-header">
                      <div>
                        <h3>{{ getGroupName(request.group_id) }}</h3>
                        <p class="request-date">{{ formatDate(request.created_at) }}</p>
                      </div>
                      <mat-chip [class]="'status-' + request.status">
                        {{ getStatusLabel(request.status) }}
                      </mat-chip>
                    </div>
                    <div *ngIf="request.reviewed_by" class="request-review">
                      Рассмотрено: {{ request.reviewed_by }} ({{ formatDate(request.reviewed_at) }})
                    </div>
                  </mat-card-content>
                </mat-card>
              </div>
            </div>
          </mat-tab>

          <!-- Tab 4: Заявки мне (Teacher) -->
          <mat-tab label="Заявки мне" *ngIf="currentUser?.role === 'teacher'">
            <div class="tab-content">
              <div *ngIf="loadingIncoming" class="loading">
                <mat-spinner></mat-spinner>
              </div>

              <div *ngIf="!loadingIncoming && allIncomingRequests.length === 0" class="empty-state">
                <mat-icon>person_add_disabled</mat-icon>
                <p>У вас нет входящих заявок</p>
              </div>

              <div class="requests-list" *ngIf="!loadingIncoming && allIncomingRequests.length > 0">
                <mat-card *ngFor="let request of allIncomingRequests" class="request-card highlight-card">
                  <mat-card-content class="request-card-content">
                    <div class="request-user-info">
                      <div class="avatar-sm">
                        <mat-icon>person</mat-icon>
                      </div>
                      <div class="user-meta">
                        <span class="user-name">{{ request.user_name }}</span>
                        <span class="group-target">в группу <b>{{ getGroupName(request.group_id) }}</b></span>
                        <span class="request-date">{{ formatDate(request.created_at) }}</span>
                      </div>
                    </div>
                    <div class="request-actions">
                      <button mat-raised-button color="primary" (click)="approveRequest(request)">Принять</button>
                      <button mat-button color="warn" (click)="rejectRequest(request)">Отклонить</button>
                      <button mat-icon-button (click)="viewGroupDetails(getGroupById(request.group_id))" matTooltip="Перейти к группе">
                        <mat-icon>arrow_forward</mat-icon>
                      </button>
                    </div>
                  </mat-card-content>
                </mat-card>
              </div>
            </div>
          </mat-tab>
        </mat-tab-group>
      </div>
    </div>
  `,
  styles: [`
    .groups-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .groups-content {
      max-width: 1400px;
      margin: 0 auto;
    }

    .page-header {
      margin-bottom: 32px;
    }

    .page-title {
      font-size: 32px;
      font-weight: 600;
      margin: 0 0 8px 0;
      color: #1a237e;
      line-height: 1.2;
    }

    .page-subtitle {
      font-size: 16px;
      color: #616161;
      margin: 0;
      line-height: 1.5;
    }

    .tab-content {
      padding: 24px 0;
    }

    .filter-card {
      margin-bottom: 24px;
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      background: white;
    }

    .filter-card mat-card-content {
      display: flex;
      gap: 16px;
      align-items: center;
    }

    .create-button {
      margin-left: auto;
    }

    .loading {
      display: flex;
      justify-content: center;
      padding: 60px;
    }

    .groups-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 24px;
    }

    .group-card {
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: all 0.3s ease;
      background: white;
    }

    .group-card:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
      transform: translateY(-4px);
    }

    .group-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      width: 100%;
    }

    .group-actions {
      display: flex;
      gap: 8px;
    }

    .group-description {
      color: #666;
      margin: 12px 0;
      line-height: 1.5;
    }

    .group-stats {
      display: flex;
      gap: 16px;
      margin-top: 12px;
    }

    .stat-item {
      display: flex;
      align-items: center;
      gap: 4px;
      color: #666;
      font-size: 14px;
    }

    .stat-item mat-icon {
      font-size: 18px;
      width: 18px;
      height: 18px;
    }

    .group-actions-bottom {
      margin-top: 16px;
      display: flex;
      justify-content: flex-end;
    }

    .empty-state {
      text-align: center;
      padding: 80px 20px;
      color: #999;
    }

    .empty-state mat-icon {
      font-size: 96px;
      width: 96px;
      height: 96px;
      margin-bottom: 24px;
      opacity: 0.4;
      color: #9e9e9e;
    }

    .empty-state p {
      font-size: 24px;
      font-weight: 500;
      color: #616161;
      margin: 0 0 8px 0;
    }

    .empty-hint {
      font-size: 16px !important;
      color: #9e9e9e;
      margin: 0;
    }

    .requests-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .request-card {
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      background: white;
    }

    .request-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .request-header h3 {
      margin: 0 0 4px 0;
      font-size: 18px;
      font-weight: 500;
    }

    .request-date {
      margin: 0;
      color: #999;
      font-size: 14px;
    }

    .request-review {
      margin-top: 12px;
      color: #666;
      font-size: 14px;
    }

    .status-pending {
      background-color: #ff9800;
      color: white;
    }

    .status-approved {
      background-color: #4caf50;
      color: white;
    }

    .status-rejected {
      background-color: #f44336;
      color: white;
    }

    .highlight-card {
      border-left: 4px solid #1a237e;
      background: #f8f9ff !important;
    }

    .request-card-content {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px !important;
    }

    .request-user-info {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .avatar-sm {
      width: 40px;
      height: 40px;
      background: #e8eaf6;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #1a237e;
    }

    .user-meta {
      display: flex;
      flex-direction: column;
    }

    .user-name {
      font-weight: 600;
      font-size: 16px;
      color: #1a237e;
    }

    .group-target {
      font-size: 14px;
      color: #666;
    }

    .request-actions {
      display: flex;
      gap: 8px;
    }

    @media (max-width: 768px) {
      .groups-container {
        padding: 16px;
      }

      .groups-grid {
        grid-template-columns: 1fr;
      }
    }
  `]
})
export class GroupsComponent implements OnInit {
  subjects: any[] = [];
  myGroups: any[] = [];
  allGroups: any[] = [];
  myRequests: any[] = [];
  allIncomingRequests: any[] = [];
  selectedSubjectId: string | null = null;
  selectedSubjectIdForAll: string | null = null;
  loading = false;
  loadingIncoming = false;
  currentUser: any = null;

  constructor(
    private apiService: ApiService,
    private authService: AuthService,
    private dialog: MatDialog,
    private fb: FormBuilder,
    private router: Router
  ) { }

  ngOnInit() {
    this.currentUser = this.authService.getCurrentUser();
    this.loadSubjects();
    this.loadMyGroups();
    this.loadAllGroups();
    if (this.currentUser) {
      this.loadMyRequests();
    }
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  loadMyGroups() {
    this.loading = true;
    this.apiService.getGroups(this.selectedSubjectId || undefined).subscribe({
      next: (groups) => {
        // Filter groups created by current user
        this.myGroups = groups.filter(g => g.created_by === this.currentUser?.name);
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading my groups:', err);
        this.loading = false;
      }
    });
  }

  loadAllGroups() {
    this.loading = true;
    this.apiService.getGroups(this.selectedSubjectIdForAll || undefined).subscribe({
      next: (groups) => {
        this.allGroups = groups;
        this.loading = false;
        if (this.currentUser?.role === 'teacher') {
          this.loadAllIncomingRequests();
        }
      },
      error: (err) => {
        console.error('Error loading all groups:', err);
        this.loading = false;
      }
    });
  }

  loadAllIncomingRequests() {
    this.loadingIncoming = true;
    this.allIncomingRequests = [];

    // We only care about groups made by current user
    const teacherGroups = this.allGroups.filter(g => g.created_by === this.currentUser?.name);

    if (teacherGroups.length === 0) {
      this.loadingIncoming = false;
      return;
    }

    let loadedCount = 0;
    teacherGroups.forEach(group => {
      this.apiService.getGroupRequests(group.id, 'pending').subscribe({
        next: (requests) => {
          this.allIncomingRequests = [...this.allIncomingRequests, ...requests];
          loadedCount++;
          if (loadedCount === teacherGroups.length) {
            this.loadingIncoming = false;
          }
        },
        error: () => {
          loadedCount++;
          if (loadedCount === teacherGroups.length) {
            this.loadingIncoming = false;
          }
        }
      });
    });
  }

  getGroupById(id: string) {
    return this.allGroups.find(g => g.id === id);
  }

  navigateToGroup(id: string) {
    this.router.navigate(['/groups', id]);
  }

  approveRequest(request: any) {
    this.apiService.updateGroupRequest(request.group_id, request.id, 'approved', this.currentUser.name).subscribe({
      next: () => {
        this.loadAllIncomingRequests();
        this.loadMyGroups();
      }
    });
  }

  rejectRequest(request: any) {
    this.apiService.updateGroupRequest(request.group_id, request.id, 'rejected', this.currentUser.name).subscribe({
      next: () => this.loadAllIncomingRequests()
    });
  }

  loadMyRequests() {
    if (!this.currentUser) return;
    this.apiService.getMyGroupRequests(this.currentUser.name).subscribe({
      next: (requests) => {
        this.myRequests = requests;
      },
      error: (err) => console.error('Error loading my requests:', err)
    });
  }

  getSubjectName(subjectId: string): string {
    const subject = this.subjects.find(s => s.id === subjectId);
    return subject ? subject.name : 'Неизвестный курс';
  }

  getGroupName(groupId: string): string {
    const group = [...this.myGroups, ...this.allGroups].find(g => g.id === groupId);
    return group ? group.name : 'Неизвестная группа';
  }

  getStatusLabel(status: string): string {
    const labels: { [key: string]: string } = {
      'pending': 'Ожидает рассмотрения',
      'approved': 'Одобрена',
      'rejected': 'Отклонена'
    };
    return labels[status] || status;
  }

  formatDate(dateString: string): string {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  }

  isRequestPending(groupId: string): boolean {
    return this.myRequests.some(r => r.group_id === groupId && r.status === 'pending');
  }

  isGroupMember(groupId: string): boolean {
    // This would need to check if user is a member - for now return false
    // TODO: Implement member check
    return false;
  }

  openCreateGroupDialog() {
    const dialogRef = this.dialog.open(CreateGroupDialogComponent, {
      width: '600px',
      data: { subjects: this.subjects, currentUser: this.currentUser }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.apiService.createGroup(result).subscribe({
          next: () => {
            this.loadMyGroups();
            this.loadAllGroups();
          },
          error: (err) => {
            console.error('Error creating group:', err);
            alert('Ошибка при создании группы: ' + (err.error?.detail || err.message));
          }
        });
      }
    });
  }

  viewGroupDetails(group: any) {
    const dialogRef = this.dialog.open(GroupDetailsDialogComponent, {
      width: '800px',
      data: { group, currentUser: this.currentUser }
    });

    dialogRef.afterClosed().subscribe(() => {
      this.loadMyGroups();
      this.loadAllGroups();
      this.loadMyRequests();
    });
  }

  deleteGroup(groupId: string) {
    if (confirm('Удалить группу? Все участники будут удалены из группы.')) {
      this.apiService.deleteGroup(groupId).subscribe({
        next: () => {
          this.loadMyGroups();
          this.loadAllGroups();
        },
        error: (err) => {
          console.error('Error deleting group:', err);
          alert('Ошибка при удалении группы');
        }
      });
    }
  }

  requestToJoinGroup(group: any) {
    if (!this.currentUser) return;

    this.apiService.createGroupRequest(group.id, this.currentUser.name).subscribe({
      next: () => {
        alert('Заявка на вступление отправлена');
        this.loadMyRequests();
      },
      error: (err) => {
        console.error('Error creating group request:', err);
        alert('Ошибка при отправке заявки: ' + (err.error?.detail || err.message));
      }
    });
  }
}

@Component({
  selector: 'app-create-group-dialog',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatSelectModule,
    MatIconModule
  ],
  template: `
    <h2 mat-dialog-title>Создать группу</h2>
    <mat-dialog-content>
      <form [formGroup]="groupForm">
        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Курс</mat-label>
          <mat-select formControlName="subject_id" required>
            <mat-option *ngFor="let subject of data.subjects" [value]="subject.id">
              {{ subject.name }}
            </mat-option>
          </mat-select>
        </mat-form-field>

        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Название группы</mat-label>
          <input matInput formControlName="name" required>
        </mat-form-field>

        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Описание (опционально)</mat-label>
          <textarea matInput formControlName="description" rows="3"></textarea>
        </mat-form-field>

        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Максимальное количество участников (опционально)</mat-label>
          <input matInput type="number" formControlName="max_size" min="1">
          <mat-hint>Оставьте пустым, если ограничения нет</mat-hint>
        </mat-form-field>
      </form>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="cancel()">Отмена</button>
      <button mat-raised-button color="primary" (click)="save()" [disabled]="!groupForm.valid">
        Создать
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width {
      width: 100%;
      margin-bottom: 16px;
    }
    mat-dialog-content {
      min-width: 500px;
      padding-top: 16px;
    }
    mat-dialog-actions {
      padding: 16px 24px;
    }
  `]
})
export class CreateGroupDialogComponent {
  groupForm: FormGroup;

  constructor(
    private fb: FormBuilder,
    private dialogRef: MatDialogRef<CreateGroupDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any
  ) {
    this.groupForm = this.fb.group({
      subject_id: ['', Validators.required],
      name: ['', Validators.required],
      description: [''],
      max_size: [null]
    });
  }

  save() {
    if (this.groupForm.valid) {
      const formValue = this.groupForm.value;
      const group = {
        subject_id: formValue.subject_id,
        name: formValue.name,
        description: formValue.description || null,
        max_size: formValue.max_size ? parseInt(formValue.max_size) : null,
        created_by: this.data.currentUser?.name || null
      };
      this.dialogRef.close(group);
    }
  }

  cancel() {
    this.dialogRef.close();
  }
}

@Component({
  selector: 'app-group-details-dialog',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatTableModule,
    MatChipsModule,
    MatTooltipModule,
    MatCardModule
  ],
  template: `
    <h2 mat-dialog-title>{{ data.group.name }}</h2>
    <mat-dialog-content>
      <div class="group-info">
        <p><strong>Курс:</strong> {{ getSubjectName(data.group.subject_id) }}</p>
        <p *ngIf="data.group.description"><strong>Описание:</strong> {{ data.group.description }}</p>
        <p><strong>Участников:</strong> {{ data.group.member_count || 0 }} / {{ data.group.max_size || '∞' }}</p>
      </div>

      <div class="section-header">
        <h3>Участники</h3>
        <button 
          *ngIf="isGroupOwner()" 
          mat-raised-button 
          color="primary" 
          (click)="openAddMemberDialog()">
          <mat-icon>person_add</mat-icon>
          Добавить участника
        </button>
      </div>

      <div *ngIf="members.length === 0" class="empty-members">
        Нет участников
      </div>

      <table mat-table [dataSource]="members" *ngIf="members.length > 0" class="members-table">
        <ng-container matColumnDef="user_name">
          <th mat-header-cell *matHeaderCellDef>Имя</th>
          <td mat-cell *matCellDef="let member">{{ member.user_name }}</td>
        </ng-container>
        <ng-container matColumnDef="joined_at">
          <th mat-header-cell *matHeaderCellDef>Дата вступления</th>
          <td mat-cell *matCellDef="let member">{{ formatDate(member.joined_at) }}</td>
        </ng-container>
        <ng-container matColumnDef="actions">
          <th mat-header-cell *matHeaderCellDef>Действия</th>
          <td mat-cell *matCellDef="let member">
            <button 
              *ngIf="isGroupOwner()" 
              mat-icon-button 
              color="warn" 
              (click)="removeMember(member.id)"
              matTooltip="Удалить из группы">
              <mat-icon>delete</mat-icon>
            </button>
          </td>
        </ng-container>
        <tr mat-header-row *matHeaderRowDef="memberColumns"></tr>
        <tr mat-row *matRowDef="let row; columns: memberColumns;"></tr>
      </table>

      <div class="section-header" style="margin-top: 32px;">
        <h3>Заявки на вступление</h3>
      </div>

      <div *ngIf="pendingRequests.length === 0" class="empty-requests">
        Нет заявок на рассмотрение
      </div>

      <div class="requests-list" *ngIf="pendingRequests.length > 0">
        <mat-card *ngFor="let request of pendingRequests" class="request-card">
          <mat-card-content>
            <div class="request-header">
              <div>
                <h4>{{ request.user_name }}</h4>
                <p class="request-date">{{ formatDate(request.created_at) }}</p>
              </div>
              <div class="request-actions" *ngIf="isGroupOwner()">
                <button mat-raised-button color="primary" (click)="approveRequest(request.id)">
                  <mat-icon>check</mat-icon>
                  Принять
                </button>
                <button mat-raised-button color="warn" (click)="rejectRequest(request.id)">
                  <mat-icon>close</mat-icon>
                  Отклонить
                </button>
              </div>
            </div>
          </mat-card-content>
        </mat-card>
      </div>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="close()">Закрыть</button>
    </mat-dialog-actions>
  `,
  styles: [`
    .group-info {
      margin-bottom: 24px;
      padding: 16px;
      background: #f5f5f5;
      border-radius: 8px;
    }

    .group-info p {
      margin: 8px 0;
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 24px 0 16px 0;
    }

    .section-header h3 {
      margin: 0;
      font-size: 20px;
      font-weight: 500;
    }

    .empty-members, .empty-requests {
      text-align: center;
      padding: 40px;
      color: #999;
    }

    .members-table {
      width: 100%;
    }

    .requests-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .request-card {
      border-radius: 12px;
    }

    .request-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .request-header h4 {
      margin: 0 0 4px 0;
      font-size: 16px;
      font-weight: 500;
    }

    .request-date {
      margin: 0;
      color: #999;
      font-size: 12px;
    }

    .request-actions {
      display: flex;
      gap: 8px;
    }

    mat-dialog-content {
      max-height: 70vh;
      overflow-y: auto;
    }
  `]
})
export class GroupDetailsDialogComponent {
  members: any[] = [];
  pendingRequests: any[] = [];
  memberColumns = ['user_name', 'joined_at', 'actions'];

  constructor(
    private dialogRef: MatDialogRef<GroupDetailsDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private apiService: ApiService,
    private dialog: MatDialog
  ) {
    this.loadMembers();
    this.loadRequests();
  }

  loadMembers() {
    this.apiService.getGroupMembers(this.data.group.id).subscribe({
      next: (members) => {
        this.members = members;
      },
      error: (err) => console.error('Error loading members:', err)
    });
  }

  loadRequests() {
    this.apiService.getGroupRequests(this.data.group.id, 'pending').subscribe({
      next: (requests) => {
        this.pendingRequests = requests;
      },
      error: (err) => console.error('Error loading requests:', err)
    });
  }

  getSubjectName(subjectId: string): string {
    // This would need subjects list - simplified for now
    return 'Курс';
  }

  formatDate(dateString: string): string {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' });
    } catch {
      return '';
    }
  }

  isGroupOwner(): boolean {
    return this.data.group.created_by === this.data.currentUser?.name;
  }

  openAddMemberDialog() {
    const dialogRef = this.dialog.open(AddMemberDialogComponent, {
      width: '500px',
      data: { groupId: this.data.group.id }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.loadMembers();
      }
    });
  }

  removeMember(memberId: string) {
    if (confirm('Удалить участника из группы?')) {
      this.apiService.removeGroupMember(this.data.group.id, memberId).subscribe({
        next: () => {
          this.loadMembers();
        },
        error: (err) => {
          console.error('Error removing member:', err);
          alert('Ошибка при удалении участника');
        }
      });
    }
  }

  approveRequest(requestId: string) {
    this.apiService.updateGroupRequest(
      this.data.group.id,
      requestId,
      'approved',
      this.data.currentUser?.name || ''
    ).subscribe({
      next: () => {
        this.loadRequests();
        this.loadMembers();
      },
      error: (err) => {
        console.error('Error approving request:', err);
        alert('Ошибка при одобрении заявки: ' + (err.error?.detail || err.message));
      }
    });
  }

  rejectRequest(requestId: string) {
    this.apiService.updateGroupRequest(
      this.data.group.id,
      requestId,
      'rejected',
      this.data.currentUser?.name || ''
    ).subscribe({
      next: () => {
        this.loadRequests();
      },
      error: (err) => {
        console.error('Error rejecting request:', err);
        alert('Ошибка при отклонении заявки');
      }
    });
  }

  close() {
    this.dialogRef.close();
  }
}

@Component({
  selector: 'app-add-member-dialog',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatTableModule,
    MatProgressSpinnerModule
  ],
  template: `
    <h2 mat-dialog-title>Добавить участника</h2>
    <mat-dialog-content>
      <mat-form-field appearance="outline" class="full-width">
        <mat-label>Поиск студентов</mat-label>
        <input matInput [(ngModel)]="searchQuery" (input)="searchUsers()" placeholder="Введите имя студента">
        <mat-icon matSuffix>search</mat-icon>
      </mat-form-field>

      <div *ngIf="searching" class="loading">
        <mat-spinner diameter="30"></mat-spinner>
      </div>

      <div *ngIf="!searching && searchResults.length === 0 && searchQuery" class="empty">
        Студенты не найдены
      </div>

      <div *ngIf="!searching && searchResults.length > 0" class="users-list">
        <div *ngFor="let user of searchResults" class="user-item" (click)="selectUser(user)">
          <mat-icon>person</mat-icon>
          <span>{{ user.name }}</span>
        </div>
      </div>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="close()">Отмена</button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width {
      width: 100%;
      margin-bottom: 16px;
    }

    .loading {
      display: flex;
      justify-content: center;
      padding: 20px;
    }

    .empty {
      text-align: center;
      padding: 40px;
      color: #999;
    }

    .users-list {
      max-height: 300px;
      overflow-y: auto;
    }

    .user-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      margin-bottom: 8px;
      cursor: pointer;
      transition: background-color 0.2s;
    }

    .user-item:hover {
      background-color: #f5f5f5;
    }

    .user-item mat-icon {
      color: #666;
    }

    mat-dialog-content {
      min-width: 400px;
      padding-top: 16px;
    }
  `]
})
export class AddMemberDialogComponent {
  searchQuery: string = '';
  searchResults: any[] = [];
  searching = false;

  constructor(
    private dialogRef: MatDialogRef<AddMemberDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private apiService: ApiService
  ) { }

  searchUsers() {
    if (!this.searchQuery || this.searchQuery.trim() === '') {
      // If search is empty, show all users
      this.apiService.getUsers().subscribe({
        next: (users) => {
          this.searchResults = users;
        },
        error: (err) => console.error('Error loading users:', err)
      });
      return;
    }

    this.searching = true;
    this.apiService.getUsers(this.searchQuery).subscribe({
      next: (users) => {
        this.searchResults = users;
        this.searching = false;
      },
      error: (err) => {
        console.error('Error searching users:', err);
        this.searching = false;
      }
    });
  }

  selectUser(user: any) {
    this.apiService.addGroupMember(this.data.groupId, { user_name: user.name }).subscribe({
      next: () => {
        this.dialogRef.close({ success: true });
      },
      error: (err: any) => {
        console.error('Error adding member:', err);
        alert('Ошибка при добавлении участника: ' + (err.error?.detail || err.message));
      }
    });
  }

  close() {
    this.dialogRef.close();
  }
}

