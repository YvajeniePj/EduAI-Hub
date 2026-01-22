import { Component, OnInit, Inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { CommonModule } from '@angular/common';
import { RouterModule, Router } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-stream-list',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatDialogModule,
    MatFormFieldModule,
    MatSelectModule,
    FormsModule
  ],
  template: `
    <div class="page-container">
      <div class="header">
        <h1>Прямые эфиры</h1>
        <button mat-raised-button color="primary" (click)="openStartStreamDialog()">
          <mat-icon>add_circle</mat-icon>
            Начать трансляцию
        </button>
      </div>

      <div class="content">
        <div *ngIf="loading" class="loading">
          <mat-spinner diameter="40"></mat-spinner>
        </div>

        <div *ngIf="!loading && activeStreams.length === 0" class="empty-state">
          <mat-icon>videocam_off</mat-icon>
          <p>В данный момент нет активных трансляций</p>
          <button mat-stroked-button color="primary" (click)="openStartStreamDialog()">
            Станьте первым, кто начнет эфир!
          </button>
        </div>

        <div class="streams-grid" *ngIf="!loading && activeStreams.length > 0">
          <mat-card *ngFor="let stream of activeStreams" class="stream-card">
            <div class="stream-preview">
              <div class="live-badge">LIVE</div>
              <mat-icon class="preview-icon">play_circle_filled</mat-icon>
            </div>
            <mat-card-content>
              <h3 class="subject-name">{{ getSubjectName(stream.subject_id) }}</h3>
              <p class="teacher-name">
                <mat-icon>person</mat-icon>
                Ведущий: {{ stream.teacher_name }}
              </p>
              <p class="start-time">
                <mat-icon>schedule</mat-icon>
                Начато: {{ stream.created_at | date:'HH:mm' }}
              </p>
            </mat-card-content>
            <mat-card-actions>
              <button mat-flat-button color="warn" [routerLink]="['/courses', stream.subject_id, 'stream']">
                Присоединиться
              </button>
            </mat-card-actions>
          </mat-card>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .page-container {
      padding: 32px;
      min-height: 100vh;
      background: #f5f5f5;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 32px;
    }

    h1 {
      font-size: 28px;
      font-weight: 600;
      color: #6200ee;
      margin: 0;
    }

    .loading {
      display: flex;
      justify-content: center;
      padding: 48px;
    }

    .empty-state {
      text-align: center;
      padding: 64px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    .empty-state mat-icon {
      font-size: 64px;
      width: 64px;
      height: 64px;
      color: #757575;
      margin-bottom: 16px;
    }

    .streams-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 24px;
    }

    .stream-card {
      border-radius: 12px;
      overflow: hidden;
      transition: transform 0.2s;
    }

    .stream-card:hover {
      transform: translateY(-4px);
    }

    .stream-preview {
      height: 160px;
      background: #000;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .live-badge {
      position: absolute;
      top: 12px;
      left: 12px;
      background: #f44336;
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: bold;
      letter-spacing: 1px;
    }

    .preview-icon {
      font-size: 48px;
      width: 48px;
      height: 48px;
      color: rgba(255,255,255,0.7);
    }

    .subject-name {
      margin: 16px 0 8px;
      font-size: 18px;
      font-weight: 600;
    }

    .teacher-name, .start-time {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #666;
      font-size: 14px;
      margin: 4px 0;
    }

    .teacher-name mat-icon, .start-time mat-icon {
      font-size: 16px;
      width: 16px;
      height: 16px;
    }

    mat-card-actions {
      padding: 16px !important;
    }

    button {
      width: 100%;
    }
  `]
})
export class StreamListComponent implements OnInit {
  activeStreams: any[] = [];
  subjects: any[] = [];
  loading = true;

  constructor(
    private apiService: ApiService,
    private dialog: MatDialog,
    private router: Router
  ) { }

  ngOnInit() {
    this.refresh();
  }

  async refresh() {
    this.loading = true;
    this.activeStreams = []; // Clear current list to avoid ghosts
    try {
      // Load both streams and subjects to show course names
      const [streams, subjects] = await Promise.all([
        this.apiService.getActiveStreamingRooms().toPromise(),
        this.apiService.getSubjects().toPromise()
      ]);

      this.activeStreams = (streams || []).filter((s: any) => s && s.subject_id);
      this.subjects = subjects || [];
    } catch (err) {
      console.error('Error refreshing streams:', err);
    } finally {
      this.loading = false;
    }
  }

  getSubjectName(subjectId: string): string {
    if (!subjectId) return 'Неизвестный курс';
    const subject = this.subjects.find(s => s && s.id === subjectId);
    return subject ? subject.name : 'Неизвестный курс';
  }

  openStartStreamDialog() {
    const dialogRef = this.dialog.open(StartStreamDialogComponent, {
      width: '400px',
      data: { subjects: this.subjects }
    });

    dialogRef.afterClosed().subscribe(subjectId => {
      if (subjectId) {
        this.router.navigate(['/courses', subjectId, 'stream']);
      }
    });
  }
}

@Component({
  selector: 'app-start-stream-dialog',
  standalone: true,
  imports: [CommonModule, MatDialogModule, MatFormFieldModule, MatSelectModule, MatButtonModule, FormsModule],
  template: `
    <h2 mat-dialog-title>Начать новый эфир</h2>
    <mat-dialog-content>
      <p>Выберите курс, по которому будет проходить трансляция:</p>
      <mat-form-field appearance="outline" class="full-width">
        <mat-label>Курс</mat-label>
        <mat-select [(ngModel)]="selectedSubjectId">
          <mat-option *ngFor="let s of subjects" [value]="s.id">
            {{ s.name }}
          </mat-option>
        </mat-select>
      </mat-form-field>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button [mat-dialog-close]="null">Отмена</button>
      <button mat-raised-button color="primary" [mat-dialog-close]="selectedSubjectId" [disabled]="!selectedSubjectId">
        Начать
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width { width: 100%; margin-top: 16px; }
  `]
})
export class StartStreamDialogComponent {
  subjects: any[] = [];
  selectedSubjectId: string = '';

  constructor(@Inject(MAT_DIALOG_DATA) public data: any) {
    this.subjects = data.subjects;
  }
}

