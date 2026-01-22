import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatDialogRef, MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-upload-material-dialog',
  standalone: true,
  imports: [
    CommonModule,
    MatDialogModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatSelectModule,
    FormsModule,
    ReactiveFormsModule
  ],
  template: `
    <h2 mat-dialog-title>Загрузить материал</h2>
    <mat-dialog-content>
      <form [formGroup]="uploadForm">
        <div class="file-upload-area">
            <input type="file" #fileInput (change)="onFileSelected($event)" multiple class="file-input" id="fileInputDlg">
            <label for="fileInputDlg" class="file-label">
              <mat-icon>attach_file</mat-icon>
              <span>{{ selectedFiles.length > 0 ? selectedFiles.length + ' файл(ов) выбрано' : 'Выберите файлы' }}</span>
            </label>
        </div>

        <mat-form-field appearance="outline" class="full-width">
          <mat-label>Описание (опционально)</mat-label>
          <input matInput formControlName="note" placeholder="Например: лекция 1, слайды">
        </mat-form-field>

        <div class="form-row" *ngIf="availableGroups.length > 0">
              <mat-form-field appearance="outline" class="full-width">
                <mat-label>Назначить группам (опционально)</mat-label>
                <mat-select formControlName="allowed_groups" multiple>
                  <mat-option *ngFor="let group of availableGroups" [value]="group.id">
                    {{ group.name }}
                  </mat-option>
                </mat-select>
                <mat-hint>Если не выбрано, материал доступен всем</mat-hint>
              </mat-form-field>
        </div>

      </form>

      <div *ngIf="loading" class="loading-indicator">
        <p>Загрузка...</p>
      </div>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="cancel()">Отмена</button>
      <button mat-raised-button color="primary" (click)="upload()" [disabled]="selectedFiles.length === 0 || loading">
        Загрузить
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width {
      width: 100%;
      margin-top: 16px;
    }
    .file-upload-area {
        margin-bottom: 16px;
    }
    .file-input {
      display: none;
    }
    .file-label {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 16px;
      border: 2px dashed #ddd;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.3s ease;
      background: #fafafa;
      justify-content: center;
    }
    .file-label:hover {
      border-color: #3f51b5;
      background: #f3f4ff;
    }
    .loading-indicator {
        text-align: center;
        margin-top: 10px;
        color: #666;
    }
  `]
})
export class UploadMaterialDialogComponent {
  uploadForm: FormGroup;
  selectedFiles: File[] = [];
  loading = false;
  availableGroups: any[] = [];

  constructor(
    private fb: FormBuilder,
    private apiService: ApiService,
    private dialogRef: MatDialogRef<UploadMaterialDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { subjectId: string }
  ) {
    this.uploadForm = this.fb.group({
      note: [''],
      allowed_groups: [[]]
    });

    this.loadGroups();
  }

  loadGroups() {
    if (this.data.subjectId) {
      this.apiService.getGroups(this.data.subjectId).subscribe({
        next: (groups) => this.availableGroups = groups,
        error: (err) => console.error('Error loading groups', err)
      });
    }
  }

  onFileSelected(event: any) {
    this.selectedFiles = Array.from(event.target.files);
  }

  upload() {
    if (this.selectedFiles.length === 0) return;

    this.loading = true;
    const note = this.uploadForm.get('note')?.value;
    const allowedGroups = this.uploadForm.get('allowed_groups')?.value;
    const subjectId = this.data.subjectId;

    let uploadCount = 0;
    let errors = 0;

    this.selectedFiles.forEach((file) => {
      const fileFormData = new FormData();
      fileFormData.append('file', file);
      fileFormData.append('subject_id', subjectId);
      // uploader is handled by API Gateway

      if (note) {
        fileFormData.append('note', note);
      }

      if (allowedGroups && allowedGroups.length > 0) {
        fileFormData.append('allowed_groups', JSON.stringify(allowedGroups));
      }

      this.apiService.uploadMaterial(fileFormData).subscribe({
        next: () => {
          uploadCount++;
          this.checkCompletion(uploadCount, errors);
        },
        error: (err) => {
          console.error('Error uploading file:', err);
          errors++;
          uploadCount++; // Processed
          this.checkCompletion(uploadCount, errors);
        }
      });
    });
  }

  checkCompletion(count: number, errors: number) {
    if (count === this.selectedFiles.length) {
      this.loading = false;
      if (errors === 0) {
        this.dialogRef.close(true);
      } else {
        alert(`Загружено файлов: ${count - errors}. Ошибок: ${errors}.`);
        this.dialogRef.close(true); // Close anyway to refresh list
      }
    }
  }

  cancel() {
    this.dialogRef.close();
  }
}
