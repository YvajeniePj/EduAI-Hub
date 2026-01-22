import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-submissions',
  standalone: true,
  imports: [CommonModule, MatCardModule],
  template: `
    <h1>Мои сдачи</h1>
    <div *ngFor="let submission of submissions">
      <mat-card>
        <mat-card-content>
          <p>Тест ID: {{ submission.test_id }}</p>
          <p>Оценка: {{ submission.total_score }}/{{ submission.total_max }}</p>
        </mat-card-content>
      </mat-card>
    </div>
  `
})
export class SubmissionsComponent implements OnInit {
  submissions: any[] = [];

  constructor(private apiService: ApiService, private auth: AuthService) {}

  ngOnInit() {
    this.loadSubmissions();
  }

  loadSubmissions() {
    const user = this.auth.getCurrentUser();
    const username = user?.name;
    this.apiService.getSubmissions(undefined, username || undefined).subscribe({
      next: (submissions) => this.submissions = submissions,
      error: (err) => console.error('Error loading submissions:', err)
    });
  }
}

