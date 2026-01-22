import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, Validators, ReactiveFormsModule, FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatDialog, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTabsModule } from '@angular/material/tabs';
import { MatMenuModule } from '@angular/material/menu';
import { RouterModule } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-generate-course-dialog',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatIconModule
  ],
  template: `
    <h2 mat-dialog-title>Сгенерировать курс с AI</h2>
    <mat-dialog-content>
      <form [formGroup]="form" *ngIf="!loading; else loadingTpl">
        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Тема курса</mat-label>
          <input matInput formControlName="topic" required placeholder="Например: Основы Python">
        </mat-form-field>

        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Дополнительные пожелания</mat-label>
          <textarea matInput formControlName="additionalInfo" rows="4" placeholder="Например: акцент на практику..."></textarea>
        </mat-form-field>

        <p class="hint">
          <mat-icon inline>auto_awesome</mat-icon>
          AI создаст структуру курса, модули, уроки и наполнит их контентом. Это может занять около 1-2 минут.
        </p>
      </form>

      <ng-template #loadingTpl>
        <div class="loading-container">
          <mat-spinner diameter="50"></mat-spinner>
          <p>AI генерирует курс...</p>
          <p class="sub-text">Пожалуйста, не закрывайте окно</p>
        </div>
      </ng-template>
    </mat-dialog-content>
    <mat-dialog-actions align="end" *ngIf="!loading">
      <button mat-button (click)="cancel()">Отмена</button>
      <button mat-raised-button color="accent" (click)="generate()" [disabled]="!form.valid">
        <mat-icon>auto_awesome</mat-icon>
        Сгенерировать
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width {
      width: 100%;
      margin-bottom: 16px;
    }
    .hint {
      font-size: 13px;
      color: #757575;
      display: flex;
      align-items: center;
      gap: 8px;
      background: #f1f8e9;
      padding: 12px;
      border-radius: 4px;
    }
    .loading-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px;
      text-align: center;
    }
    .loading-container p {
      margin-top: 16px;
      font-weight: 500;
      color: #4caf50;
    }
    .sub-text {
      font-size: 12px;
      color: #9e9e9e !important;
      margin-top: 4px !important;
    }
  `]
})
export class GenerateCourseDialogComponent {
  form: FormGroup;
  loading = false;

  constructor(
    private fb: FormBuilder,
    private dialogRef: MatDialogRef<GenerateCourseDialogComponent>,
    private apiService: ApiService
  ) {
    this.form = this.fb.group({
      topic: ['', Validators.required],
      additionalInfo: ['']
    });
  }

  generate() {
    if (this.form.valid) {
      this.loading = true;
      const { topic, additionalInfo } = this.form.value;

      this.apiService.generateCourse(topic, additionalInfo).subscribe({
        next: (res) => {
          this.loading = false;
          this.dialogRef.close(true); // Return true on success
        },
        error: (err) => {
          this.loading = false;
          console.error('Error generating course:', err);
          alert('Ошибка генерации: ' + (err.error?.detail || err.message));
        }
      });
    }
  }

  cancel() {
    this.dialogRef.close();
  }
}

@Component({
  selector: 'app-create-subject-dialog',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule
  ],
  template: `
    <h2 mat-dialog-title>Создать курс</h2>
    <mat-dialog-content>
      <form [formGroup]="form">
        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Название курса</mat-label>
          <input matInput formControlName="name" required>
        </mat-form-field>

        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Описание (опционально)</mat-label>
          <textarea matInput formControlName="description" rows="4" placeholder="Краткое описание курса"></textarea>
        </mat-form-field>
      </form>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="cancel()">Отмена</button>
      <button mat-raised-button color="primary" (click)="save()" [disabled]="!form.valid">
        Создать
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width {
      width: 100%;
      margin-bottom: 16px;
    }
  `]
})
export class CreateSubjectDialogComponent {
  form: FormGroup;

  constructor(
    private fb: FormBuilder,
    private dialogRef: MatDialogRef<CreateSubjectDialogComponent>
  ) {
    this.form = this.fb.group({
      name: ['', Validators.required],
      description: ['']
    });
  }

  save() {
    if (this.form.valid) {
      this.dialogRef.close(this.form.value);
    }
  }

  cancel() {
    this.dialogRef.close();
  }
}

@Component({
  selector: 'app-subjects',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatDialogModule,
    MatTabsModule,
    MatMenuModule,
    RouterModule,
    FormsModule
  ],
  template: `
    <div class="page-container">
      <div class="header">
        <h1>Мои курсы</h1>
        <button mat-raised-button color="warn" class="live-dashboard-btn" [routerLink]="['/streaming']">
          <mat-icon>live_tv</mat-icon>
          Прямой эфир
        </button>
      </div>

      <div class="content-wrapper">
        <mat-tab-group animationDuration="0ms" class="custom-tabs">
          <mat-tab label="Обзор курсов">
            <div class="tab-content">
              <div class="filters-bar">

                
                <div class="search-field">
                  <input type="text" placeholder="Найти" [(ngModel)]="searchQuery" (input)="filterSubjects()">
                </div>

                <div class="spacer"></div>

                <button mat-raised-button color="accent" (click)="openGenerateDialog()" class="ai-btn">
                  <mat-icon>auto_awesome</mat-icon>
                  AI Генерация
                </button>
                
                <button mat-raised-button color="primary" (click)="openCreateDialog()">
                  Создать курс
                </button>
              </div>

              <div class="courses-grid" *ngIf="subjects.length > 0">
                <mat-card *ngFor="let subject of subjects" class="course-card" [routerLink]="['/courses', subject.id]">
                  <div class="course-cover" [class.has-image]="subject.cover_image">
                    <img *ngIf="subject.cover_image" [src]="'/api/subjects/' + subject.id + '/cover'" alt="" class="cover-img">
                    <div *ngIf="!subject.cover_image" class="cover-pattern">
                      <mat-icon class="cover-icon">auto_awesome</mat-icon>
                      <div class="cover-text">{{ subject.name }}</div>
                    </div>
                  </div>
                  <mat-card-content class="course-info">
                    <div class="course-name">{{ subject.name }}</div>
                    <div class="course-description" *ngIf="subject.description">{{ subject.description | slice:0:60 }}{{ subject.description?.length > 60 ? '...' : '' }}</div>
                  </mat-card-content>
                  <div class="course-actions">
                    <button mat-icon-button class="more-btn" (click)="$event.stopPropagation();" [matMenuTriggerFor]="menu">
                      <mat-icon>more_vert</mat-icon>
                    </button>
                    <mat-menu #menu="matMenu">
                      <button mat-menu-item (click)="openCoverUpload(subject)">
                        <mat-icon>image</mat-icon>
                        <span>Загрузить обложку</span>
                      </button>
                      <button mat-menu-item [routerLink]="['/course-builder', subject.id]">
                        <mat-icon>edit</mat-icon>
                        <span>Редактировать</span>
                      </button>
                      <button mat-menu-item (click)="deleteSubject(subject.id)">
                        <mat-icon>delete</mat-icon>
                        <span>Удалить</span>
                      </button>
                    </mat-menu>
                  </div>
                </mat-card>
              </div>

              <div *ngIf="subjects.length === 0" class="empty-state">
                <p>Нет доступных курсов</p>
                <div class="button-row">
                  <button mat-raised-button color="accent" (click)="openGenerateDialog()">
                    <mat-icon>auto_awesome</mat-icon>
                    Сгенерировать с AI
                  </button>
                  <button mat-raised-button color="primary" (click)="openCreateDialog()">
                    Создать курс
                  </button>
                </div>
              </div>
            </div>
          </mat-tab>
        </mat-tab-group>
      </div>

      <!-- Floating Action Button for creating courses -->
      <button mat-fab color="primary" class="fab-add" (click)="openCreateDialog()">
        <mat-icon>add</mat-icon>
      </button>
    </div>
  `,
  styles: [`
    .page-container {
      padding: 0;
      background-color: #fff;
      min-height: 100vh;
    }

    .header {
      padding: 24px 32px 0;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .live-dashboard-btn {
      font-weight: bold;
      animation: pulse-red 2s infinite;
      height: 48px;
      padding: 0 24px;
      font-size: 16px;
    }

    @keyframes pulse-red {
      0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
      70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
      100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }

    h1 {
      font-size: 28px;
      font-weight: 600;
      color: #6200ee; /* Purple color from screenshot */
      margin: 0;
    }

    .content-wrapper {
      padding: 0 32px;
    }

    /* Custom Tabs Styling */
    ::ng-deep .custom-tabs .mat-mdc-tab-label-container {
      border-bottom: 1px solid #e0e0e0;
    }

    ::ng-deep .custom-tabs .mat-mdc-tab-label {
      font-weight: 600;
      color: #6200ee;
      opacity: 1;
      font-size: 14px;
      text-transform: uppercase;
    }

    ::ng-deep .custom-tabs .mat-mdc-tab-indicator .mdc-tab-indicator__content--underline {
      border-color: #6200ee !important;
    }

    .tab-content {
      padding-top: 24px;
    }

    /* Filters Bar */
    .filters-bar {
      display: flex;
      gap: 12px;
      margin-bottom: 32px;
      flex-wrap: wrap;
      align-items: center;
      background: #f9f9f9;
      padding: 16px;
      border: 1px solid #e0e0e0;
      border-radius: 4px;
    }

    .filter-btn, .sort-btn, .view-btn {
      color: #757575;
      border-color: #e0e0e0;
      font-weight: 400;
      background: white;
    }

    .search-field input {
      padding: 8px 12px;
      border: 1px solid #e0e0e0;
      border-radius: 4px;
      width: 200px;
      font-size: 14px;
    }

    .spacer {
      flex: 1;
    }

    .ai-btn {
      margin-right: 12px;
      background-color: #b388ff !important; /* Lighter purple accent */
      color: #311b92 !important;
    }

    /* Grid */
    .courses-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 24px;
    }

    .course-card {
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
      transition: all 0.3s cubic-bezier(.25,.8,.25,1);
      cursor: pointer;
      position: relative;
      overflow: hidden;
      border: 1px solid #e0e0e0;
    }

    .course-card:hover {
      box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22);
    }

    .course-cover {
      height: 140px;
      background-color: #1a1a1a;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
    }
    
    .cover-pattern {
        text-align: center;
    }
    
    .cover-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .course-cover.has-image {
        padding: 0;
    }

    .cover-icon {
        color: #cddc39; /* Lime green accent */
    }

    .cover-text {
        font-weight: 700;
        text-transform: uppercase;
        padding: 0 16px;
    }

    .course-info {
      padding: 16px;
    }

    .course-name {
      font-size: 16px;
      font-weight: 500;
      color: #000;
      margin-bottom: 4px;
      line-height: 1.4;
    }

    .course-description {
      font-size: 12px;
      color: #757575;
      line-height: 1.4;
    }

    .course-actions {
      position: absolute;
      bottom: 8px;
      right: 8px;
    }
    
    .more-btn {
        color: #757575;
    }

    .fab-add {
      position: fixed;
      bottom: 32px;
      right: 32px;
      background-color: #6200ee;
    }

    .empty-state {
        text-align: center;
        margin-top: 48px;
        color: #757575;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .button-row {
        display: flex;
        gap: 16px;
        margin-top: 16px;
    }
  `]
})
export class SubjectsComponent implements OnInit {
  subjects: any[] = [];
  allSubjects: any[] = [];
  searchQuery: string = '';

  constructor(
    private apiService: ApiService,
    private dialog: MatDialog
  ) { }

  ngOnInit() {
    this.loadSubjects();
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.allSubjects = subjects;
        this.filterSubjects();
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  filterSubjects() {
    if (!this.searchQuery) {
      this.subjects = [...this.allSubjects];
    } else {
      const query = this.searchQuery.toLowerCase().trim();
      this.subjects = this.allSubjects.filter(subject =>
        subject.name.toLowerCase().includes(query) ||
        (subject.description && subject.description.toLowerCase().includes(query))
      );
    }
  }

  openCreateDialog() {
    const dialogRef = this.dialog.open(CreateSubjectDialogComponent, {
      width: '500px'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.apiService.createSubject(result.name, result.description).subscribe({
          next: () => {
            this.loadSubjects();
          },
          error: (err) => {
            console.error('Error creating subject:', err);
            alert('Ошибка при создании курса: ' + (err.error?.detail || err.message));
          }
        });
      }
    });
  }

  openGenerateDialog() {
    const dialogRef = this.dialog.open(GenerateCourseDialogComponent, {
      width: '500px',
      disableClose: true // Prevent closing while generating
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.loadSubjects(); // Reload to show the new course
        alert('Курс успешно создан AI!');
      }
    });
  }

  deleteSubject(id: string) {
    if (confirm('Удалить курс?')) {
      this.apiService.deleteSubject(id).subscribe({
        next: () => this.loadSubjects(),
        error: (err) => console.error('Error deleting subject:', err)
      });
    }
  }

  openCoverUpload(subject: any) {
    // Create a hidden file input
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/jpeg,image/png,image/webp,image/gif';

    input.onchange = (e: any) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      this.apiService.uploadSubjectCover(subject.id, formData).subscribe({
        next: () => {
          this.loadSubjects();
          alert('Обложка загружена! Рекомендуемый размер: 600×400px');
        },
        error: (err) => {
          console.error('Error uploading cover:', err);
          alert('Ошибка при загрузке обложки: ' + (err.error?.detail || err.message));
        }
      });
    };

    input.click();
  }
}
