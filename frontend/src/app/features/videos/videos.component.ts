import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-videos',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatButtonModule,
    MatInputModule,
    MatSelectModule,
    MatListModule,
    MatIconModule
  ],
  template: `
    <div class="videos-container">
      <div class="page-header">
        <h1 class="page-title">Видео материалы</h1>
        <p class="page-subtitle">Добавляйте и просматривайте учебные видео</p>
      </div>
      
      <mat-card *ngIf="subjects.length > 0" class="filter-card">
        <mat-card-content>
          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Выберите предмет</mat-label>
            <mat-select [(ngModel)]="selectedSubjectId" (selectionChange)="loadVideos()">
              <mat-option *ngFor="let subject of subjects" [value]="subject.id">
                {{ subject.name }}
              </mat-option>
            </mat-select>
          </mat-form-field>
        </mat-card-content>
      </mat-card>

      <mat-card class="add-video-card">
        <mat-card-header>
          <mat-card-title>
            <mat-icon>video_library</mat-icon>
            Добавить видео
          </mat-card-title>
        </mat-card-header>
        <mat-card-content>
          <div class="add-video-form">
            <mat-form-field appearance="outline" class="full-width">
              <mat-label>URL видео (YouTube, VK)</mat-label>
              <input matInput [(ngModel)]="videoUrl" placeholder="https://www.youtube.com/watch?v=...">
              <mat-icon matPrefix>link</mat-icon>
            </mat-form-field>
            <mat-form-field appearance="outline" class="full-width">
              <mat-label>Описание (опционально)</mat-label>
              <input matInput [(ngModel)]="videoNote" placeholder="Например: лекция 1">
            </mat-form-field>
            <button 
              mat-raised-button 
              color="primary" 
              (click)="addVideo()" 
              [disabled]="!selectedSubjectId || !videoUrl"
              class="add-button">
              <mat-icon>add</mat-icon>
              Добавить видео
            </button>
          </div>
        </mat-card-content>
      </mat-card>

      <div *ngIf="videos.length > 0" class="videos-section">
        <h2 class="section-title">Видео материалы</h2>
        <div class="videos-grid">
          <mat-card *ngFor="let video of videos" class="video-card">
            <mat-card-header>
              <mat-card-title>
                <mat-icon class="video-icon">play_circle</mat-icon>
                {{ video.title }}
              </mat-card-title>
              <mat-card-subtitle *ngIf="video.note">{{ video.note }}</mat-card-subtitle>
            </mat-card-header>
            <mat-card-content>
              <div *ngIf="video.video_info?.type === 'youtube' && video.video_info?.embed_url" class="video-embed">
                <iframe 
                  [src]="getSafeUrl(video.video_info.embed_url)" 
                  frameborder="0" 
                  allowfullscreen>
                </iframe>
              </div>
              <div *ngIf="video.video_info?.type !== 'youtube'" class="video-link">
                <a [href]="video.url" target="_blank" class="external-link">
                  <mat-icon>open_in_new</mat-icon>
                  {{ video.url }}
                </a>
              </div>
            </mat-card-content>
            <mat-card-actions>
              <button mat-button color="warn" (click)="deleteVideo(video.id)" class="delete-button">
                <mat-icon>delete</mat-icon>
                Удалить
              </button>
            </mat-card-actions>
          </mat-card>
        </div>
      </div>

      <div *ngIf="videos.length === 0 && selectedSubjectId" class="empty-state">
        <mat-icon>videocam_off</mat-icon>
        <p>Нет добавленных видео</p>
      </div>
    </div>
  `,
  styles: [`
    .videos-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }

    .page-header {
      margin-bottom: 32px;
    }

    .page-title {
      font-size: 32px;
      font-weight: 600;
      margin: 0 0 8px 0;
      color: #1a1a1a;
      letter-spacing: -0.5px;
    }

    .page-subtitle {
      font-size: 16px;
      color: #666;
      margin: 0;
    }

    .filter-card, .add-video-card {
      margin-bottom: 24px;
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: box-shadow 0.3s ease;
    }

    .filter-card:hover, .add-video-card:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }

    .add-video-card mat-card-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 20px;
      font-weight: 500;
    }

    .add-video-form {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .full-width {
      width: 100%;
    }

    .add-button {
      align-self: flex-start;
      padding: 8px 24px;
    }

    .videos-section {
      margin-top: 32px;
    }

    .section-title {
      font-size: 24px;
      font-weight: 600;
      margin: 0 0 20px 0;
      color: #1a1a1a;
    }

    .videos-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
      gap: 24px;
    }

    .video-card {
      border-radius: 12px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
      transition: all 0.3s ease;
    }

    .video-card:hover {
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
      transform: translateY(-2px);
    }

    .video-icon {
      margin-right: 8px;
      color: #ff0000;
      vertical-align: middle;
    }

    .video-embed {
      position: relative;
      padding-bottom: 56.25%;
      height: 0;
      overflow: hidden;
      border-radius: 8px;
      background: #000;
      margin-bottom: 16px;
    }

    .video-embed iframe {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }

    .video-link {
      padding: 16px;
      background: #f5f5f5;
      border-radius: 8px;
    }

    .external-link {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #3f51b5;
      text-decoration: none;
      word-break: break-all;
    }

    .external-link:hover {
      text-decoration: underline;
    }

    .delete-button {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .empty-state {
      text-align: center;
      padding: 60px 20px;
      color: #999;
    }

    .empty-state mat-icon {
      font-size: 64px;
      width: 64px;
      height: 64px;
      margin-bottom: 16px;
      opacity: 0.5;
    }

    .empty-state p {
      font-size: 18px;
      margin: 0;
    }
  `]
})
export class VideosComponent implements OnInit {
  subjects: any[] = [];
  videos: any[] = [];
  selectedSubjectId: string = '';
  videoUrl: string = '';
  videoNote: string = '';

  constructor(
    private apiService: ApiService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit() {
    this.loadSubjects();
  }

  loadSubjects() {
    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        this.subjects = subjects;
        if (subjects.length > 0 && !this.selectedSubjectId) {
          this.selectedSubjectId = subjects[0].id;
          this.loadVideos();
        }
      },
      error: (err) => console.error('Error loading subjects:', err)
    });
  }

  loadVideos() {
    if (!this.selectedSubjectId) return;
    
    this.apiService.getVideos(this.selectedSubjectId).subscribe({
      next: (videos) => {
        this.videos = videos;
      },
      error: (err) => console.error('Error loading videos:', err)
    });
  }

  addVideo() {
    if (!this.selectedSubjectId || !this.videoUrl) return;

    this.apiService.createVideo({
      subject_id: this.selectedSubjectId,
      url: this.videoUrl,
      title: 'Загрузка...', // Will be updated by backend
      note: this.videoNote,
      uploader: 'current_user' // TODO: get from auth
    }).subscribe({
      next: () => {
        this.videoUrl = '';
        this.videoNote = '';
        this.loadVideos();
        alert('Видео добавлено!');
      },
      error: (err) => {
        console.error('Error adding video:', err);
        alert('Ошибка при добавлении видео: ' + (err.error?.detail || err.message));
      }
    });
  }

  deleteVideo(videoId: string) {
    if (!confirm('Удалить это видео?')) return;
    
    this.apiService.deleteVideo(videoId).subscribe({
      next: () => {
        this.loadVideos();
      },
      error: (err) => {
        console.error('Error deleting video:', err);
        alert('Ошибка при удалении: ' + (err.error?.detail || err.message));
      }
    });
  }

  getSafeUrl(url: string): SafeResourceUrl {
    return this.sanitizer.bypassSecurityTrustResourceUrl(url);
  }
}

