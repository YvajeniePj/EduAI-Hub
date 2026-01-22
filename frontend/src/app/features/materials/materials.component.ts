import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { MatExpansionModule } from '@angular/material/expansion';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-materials',
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
    MatExpansionModule
  ],
  template: `
    <div class="materials-container">
      <div class="materials-content">
        <div class="page-header">
          <h1 class="page-title">Материалы курса</h1>
          <p class="page-subtitle">Загружайте и управляйте учебными материалами</p>
        </div>
      
      <mat-card *ngIf="subjects.length > 0" class="filter-card">
        <mat-card-content>
          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Выберите курс</mat-label>
            <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="loadMaterials()">
              <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                {{ subject.name }}
              </mat-option>
            </mat-select>
          </mat-form-field>
        </mat-card-content>
      </mat-card>

      <mat-card class="upload-card">
        <mat-card-header>
          <mat-card-title>
            <mat-icon>cloud_upload</mat-icon>
            Загрузить материал
          </mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="upload-section">
            <input type="file" #fileInput (change)="onFileSelected($event)" multiple class="file-input" id="fileInput">
            <label for="fileInput" class="file-label">
              <mat-icon>attach_file</mat-icon>
              <span>{{ selectedFiles.length > 0 ? selectedFiles.length + ' файл(ов) выбрано' : 'Выберите файлы' }}</span>
            </label>
            
            <mat-form-field appearance="outline" class="full-width">
              <mat-label>Описание (опционально)</mat-label>
              <input matInput [(ngModel)]="uploadNote" placeholder="Например: лекция 1, слайды">
            </mat-form-field>
            
            <button 
              mat-raised-button 
              color="primary" 
              (click)="uploadFiles()" 
              [disabled]="!selectedSubjectId || !selectedFiles.length || loading"
              class="upload-button">
              <mat-icon>upload</mat-icon>
              {{ loading ? 'Загрузка...' : 'Загрузить' }}
            </button>
          </div>
        </mat-card-content>
      </mat-card>

      <div *ngIf="materials.length > 0" class="materials-section">
        <h2 class="section-title">Загруженные материалы</h2>
        <div class="materials-list">
          <mat-expansion-panel *ngFor="let material of materials" class="material-panel">
            <mat-expansion-panel-header>
              <mat-panel-title>
                <mat-icon class="material-icon">description</mat-icon>
                {{ material.original_name || material.name }}
              </mat-panel-title>
              <mat-panel-description>
                <span class="material-meta">{{ material.note || 'Без описания' }}</span>
                <span class="material-size">{{ formatSize(material.size) }}</span>
              </mat-panel-description>
            </mat-expansion-panel-header>
            
            <div class="material-actions">
              <button 
                *ngIf="!hasAnnotation(material)"
                mat-raised-button 
                color="primary" 
                (click)="createAnnotation(material.id)" 
                [disabled]="material.creatingAnnotation"
                class="action-button">
                <mat-icon>auto_awesome</mat-icon>
                {{ material.creatingAnnotation ? 'Создание...' : 'Создать аннотацию' }}
              </button>
              
              <div *ngIf="hasAnnotation(material)" class="annotation-section">
                <div class="language-selector">
                  <mat-form-field appearance="outline">
                    <mat-label>Язык отображения</mat-label>
                    <mat-select [(ngModel)]="material.displayLanguage" [ngModelOptions]="{standalone: true}">
                      <mat-option *ngIf="material.annotation_ru" value="ru">🇷🇺 Русский</mat-option>
                      <mat-option *ngIf="material.annotation_en" value="en">🇬🇧 English</mat-option>
                    </mat-select>
                  </mat-form-field>
                </div>
                
                <div class="annotation-box">
                  <h4 class="annotation-title">Аннотация</h4>
                  <p class="annotation-text">{{ getDisplayAnnotation(material) }}</p>
                </div>
              </div>
            </div>
            
            <div class="material-footer">
              <button mat-button color="warn" (click)="deleteMaterial(material.id)" class="delete-button">
                <mat-icon>delete</mat-icon>
                Удалить
              </button>
            </div>
          </mat-expansion-panel>
        </div>
      </div>
      
      <div *ngIf="materials.length === 0 && selectedSubjectId" class="empty-state">
        <mat-icon>folder_open</mat-icon>
        <p>Нет загруженных материалов</p>
      </div>
      </div>
    </div>
  `,
  styles: [`
    .materials-container {
      min-height: 100vh;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      padding: 24px;
    }

    .materials-content {
      max-width: 1200px;
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

    .filter-card, .upload-card {
      margin-bottom: 24px;
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: box-shadow 0.3s ease;
      background: white;
    }

    .filter-card:hover, .upload-card:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }

    .upload-card mat-card-header {
      margin-bottom: 16px;
    }

    .upload-card mat-card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 20px;
      font-weight: 500;
    }

    .upload-section {
      display: flex;
      flex-direction: column;
      gap: 16px;
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
    }

    .file-label:hover {
      border-color: #3f51b5;
      background: #f3f4ff;
    }

    .file-label mat-icon {
      color: #3f51b5;
    }

    .full-width {
      width: 100%;
    }

    .upload-button {
      align-self: flex-start;
      padding: 12px 32px;
      font-size: 16px;
      font-weight: 500;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }

    .materials-section {
      margin-top: 32px;
    }

    .section-title {
      font-size: 24px;
      font-weight: 600;
      margin: 0 0 24px 0;
      color: #1a237e;
    }

    .materials-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .material-panel {
      border-radius: 16px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: all 0.3s ease;
      background: white;
    }

    .material-panel:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
      transform: translateY(-2px);
    }

    .material-icon {
      margin-right: 8px;
      color: #3f51b5;
    }

    .material-meta {
      color: #666;
      margin-right: 12px;
    }

    .material-size {
      color: #999;
      font-size: 14px;
    }

    .material-actions {
      margin: 20px 0;
    }

    .action-button {
      margin-bottom: 16px;
    }

    .annotation-section {
      margin-top: 20px;
    }

    .language-selector {
      margin-bottom: 16px;
      max-width: 250px;
    }

    .annotation-box {
      padding: 20px;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .annotation-title {
      margin: 0 0 12px 0;
      font-size: 18px;
      font-weight: 600;
      color: #1a1a1a;
    }

    .annotation-text {
      margin: 0;
      line-height: 1.6;
      color: #333;
      white-space: pre-wrap;
    }

    .material-footer {
      display: flex;
      justify-content: flex-end;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid #eee;
    }

    .delete-button {
      display: flex;
      align-items: center;
      gap: 4px;
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
      .materials-container {
        padding: 16px;
      }
    }

    ::ng-deep .mat-expansion-panel-body {
      padding: 16px 24px 24px 24px !important;
    }
  `]
})
export class MaterialsComponent implements OnInit {
  subjects: any[] = [];
  materials: any[] = [];
  selectedSubjectId: string = '';
  selectedFiles: File[] = [];
  uploadNote: string = '';
  loading = false;

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.loadSubjects();
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
        if (subjects.length > 0 && !this.selectedSubjectId) {
          this.selectedSubjectId = subjects[0].id;
          this.loadMaterials();
        }
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  loadMaterials() {
    if (!this.selectedSubjectId) return;
    
    this.apiService.getMaterials(this.selectedSubjectId).subscribe({
      next: (materials) => {
        // Initialize display language for each material
        this.materials = materials.map((m: any) => {
          if (!m.displayLanguage) {
            // Set default display language based on available annotations
            if (m.annotation_ru) {
              m.displayLanguage = 'ru';
            } else if (m.annotation_en) {
              m.displayLanguage = 'en';
            }
          }
          return m;
        });
      },
      error: (err) => console.error('Error loading materials:', err)
    });
  }

  onFileSelected(event: any) {
    this.selectedFiles = Array.from(event.target.files);
  }

  uploadFiles() {
    if (!this.selectedSubjectId || this.selectedFiles.length === 0) return;

    this.loading = true;
    const formData = new FormData();
    
    this.selectedFiles.forEach((file, index) => {
      formData.append('file', file);
    });
    
    formData.append('subject_id', this.selectedSubjectId);
    formData.append('uploader', 'current_user'); // TODO: get from auth
    if (this.uploadNote) {
      formData.append('note', this.uploadNote);
    }

    // Upload files one by one
    let uploadCount = 0;
    this.selectedFiles.forEach((file) => {
      const fileFormData = new FormData();
      fileFormData.append('file', file);
      fileFormData.append('subject_id', this.selectedSubjectId);
      fileFormData.append('uploader', 'current_user');
      if (this.uploadNote) {
        fileFormData.append('note', this.uploadNote);
      }

      this.apiService.uploadMaterial(fileFormData).subscribe({
        next: () => {
          uploadCount++;
          if (uploadCount === this.selectedFiles.length) {
            this.loading = false;
            this.selectedFiles = [];
            this.uploadNote = '';
            this.loadMaterials();
            alert('Файлы загружены успешно!');
          }
        },
        error: (err) => {
          this.loading = false;
          console.error('Error uploading file:', err);
          alert('Ошибка при загрузке файла: ' + (err.error?.detail || err.message));
        }
      });
    });
  }

  createAnnotation(materialId: string) {
    // Find material to show loading state
    const material = this.materials.find(m => m.id === materialId);
    if (material) {
      material.creatingAnnotation = true;
    }
    
    this.apiService.createMaterialAnnotation(materialId).subscribe({
      next: (result) => {
        if (material) {
          material.creatingAnnotation = false;
          // Update annotations
          material.annotation_ru = result.annotation_ru;
          material.annotation_en = result.annotation_en;
          // Set default display language
          if (result.annotation_ru) {
            material.displayLanguage = 'ru';
          } else if (result.annotation_en) {
            material.displayLanguage = 'en';
          }
        }
        alert('Аннотации созданы на русском и английском языках!');
      },
      error: (err) => {
        if (material) {
          material.creatingAnnotation = false;
        }
        console.error('Error creating annotation:', err);
        alert('Ошибка при создании аннотации: ' + (err.error?.detail || err.message));
      }
    });
  }

  hasAnnotation(material: any): boolean {
    return !!(material.annotation_ru || material.annotation_en || material.annotation);
  }

  getDisplayAnnotation(material: any): string {
    const lang = material.displayLanguage || 'ru';
    if (lang === 'en' && material.annotation_en) {
      return material.annotation_en;
    }
    if (lang === 'ru' && material.annotation_ru) {
      return material.annotation_ru;
    }
    // Fallback to old annotation field or available language
    return material.annotation_ru || material.annotation_en || material.annotation || 'Аннотация не найдена';
  }

  deleteMaterial(materialId: string) {
    if (!confirm('Удалить этот материал?')) return;
    
    this.apiService.deleteMaterial(materialId).subscribe({
      next: () => {
        this.loadMaterials();
      },
      error: (err) => {
        console.error('Error deleting material:', err);
        alert('Ошибка при удалении: ' + (err.error?.detail || err.message));
      }
    });
  }

  formatSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }
}

