import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { UserProfileDialogComponent } from './user-profile-dialog.component';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-students',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
    MatTableModule,
    MatTableModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatDialogModule
  ],
  template: `
    <div class="students-container">
      <div class="students-content">
        <div class="page-header">
          <h1 class="page-title">Пользователи</h1>
          <p class="page-subtitle">Список всех пользователей в системе</p>
        </div>

        <mat-card class="filter-card">
          <mat-card-content>
            <div class="filters">
              <mat-form-field appearance="outline">
                <mat-label>Курс</mat-label>
                <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="loadStudents()">
                  <mat-option [value]="null">Все курсы</mat-option>
                  <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                    {{ subject.name }}
                  </mat-option>
                </mat-select>
              </mat-form-field>

              <mat-form-field appearance="outline">
                <mat-label>Поиск</mat-label>
                <input matInput [(ngModel)]="searchQuery" (input)="loadStudents()" placeholder="Введите имя студента">
                <mat-icon matSuffix>search</mat-icon>
              </mat-form-field>
            </div>
          </mat-card-content>
        </mat-card>

        <div *ngIf="loading" class="loading">
          <mat-spinner></mat-spinner>
        </div>

        <mat-card *ngIf="!loading" class="students-card">
          <mat-card-content>
            <div *ngIf="students.length === 0" class="empty-state">
              <mat-icon>people</mat-icon>
              <p>Студенты не найдены</p>
            </div>

            <table mat-table [dataSource]="students" *ngIf="students.length > 0" class="students-table">
              <ng-container matColumnDef="name">
                <th mat-header-cell *matHeaderCellDef>Имя</th>
                <td mat-cell *matCellDef="let student">
                  <span class="user-link" (click)="viewProfile(student)">{{ student.name }}</span>
                </td>
              </ng-container>

              <ng-container matColumnDef="groups">
                <th mat-header-cell *matHeaderCellDef>Группы</th>
                <td mat-cell *matCellDef="let student">
                  <div *ngIf="student.groups && student.groups.length > 0" class="groups-list">
                    <span *ngFor="let group of student.groups" class="group-chip">
                      {{ group.name }} ({{ getSubjectName(group.subject_id) }})
                    </span>
                  </div>
                  <span *ngIf="!student.groups || student.groups.length === 0" class="no-groups">
                    Нет групп
                  </span>
                </td>
              </ng-container>

              <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
              <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
            </table>
          </mat-card-content>
        </mat-card>
      </div>
    </div>
  `,
  styles: [`
    .students-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .students-content {
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

    .filter-card {
      margin-bottom: 24px;
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      background: white;
    }

    .filters {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
    }

    .loading {
      display: flex;
      justify-content: center;
      padding: 60px;
    }

    .students-card {
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      background: white;
    }

    .students-table {
      width: 100%;
    }

    .groups-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .group-chip {
      display: inline-block;
      padding: 4px 12px;
      background: #e3f2fd;
      color: #1976d2;
      border-radius: 16px;
      font-size: 14px;
    }

    .no-groups {
      color: #999;
      font-style: italic;
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
      margin: 0;
    }

    @media (max-width: 768px) {
      .students-container {
        padding: 16px;
      }

      .filters {
        grid-template-columns: 1fr;
      }
    }
    .user-link {
        color: #3f51b5;
        cursor: pointer;
        font-weight: 500;
        text-decoration: underline;
    }
    .user-link:hover {
        color: #1a237e;
    }
  `]
})
export class StudentsComponent implements OnInit {
  students: any[] = [];
  subjects: any[] = [];
  selectedSubjectId: string | null = null;
  searchQuery: string = '';
  loading = false;
  displayedColumns: string[] = ['name', 'groups'];

  constructor(
    private apiService: ApiService,
    private router: Router,
    private dialog: MatDialog
  ) { }

  ngOnInit() {
    this.loadSubjects();
    this.loadStudents();
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  loadStudents() {
    this.loading = true;
    const search = this.searchQuery.trim() || undefined;
    this.apiService.getUsers(search).subscribe({
      next: (users) => {
        // Load groups for each user
        const userPromises = users.map(user =>
          this.apiService.getGroups(undefined, user.name).toPromise().then(groups => ({
            ...user,
            groups: groups || []
          }))
        );

        Promise.all(userPromises).then(studentsWithGroups => {
          // Filter by subject if selected
          if (this.selectedSubjectId) {
            this.students = studentsWithGroups.map(student => ({
              ...student,
              groups: student.groups.filter((g: any) => g.subject_id === this.selectedSubjectId)
            })).filter(student => student.groups.length > 0);
          } else {
            this.students = studentsWithGroups;
          }

          // Sort alphabetically by name
          this.students.sort((a, b) => a.name.localeCompare(b.name));
          this.loading = false;
        });
      },
      error: (err) => {
        console.error('Error loading students:', err);
        this.loading = false;
      }
    });
  }

  getSubjectName(subjectId: string): string {
    const subject = this.subjects.find(s => s.id === subjectId);
    return subject ? subject.name : 'Неизвестный курс';
  }

  viewProfile(user: any) {
    this.dialog.open(UserProfileDialogComponent, {
      width: '400px',
      data: { user }
    });
  }
}
