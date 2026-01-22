import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatTableModule } from '@angular/material/table';
import { MatIconModule } from '@angular/material/icon';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-leaderboard',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatTableModule,
    MatIconModule
  ],
  template: `
    <div class="leaderboard-container">
      <h1>Лидерборд</h1>
      
      <mat-card>
        <mat-card-content>
          <table mat-table [dataSource]="leaderboard" class="leaderboard-table">
            <ng-container matColumnDef="rank">
              <th mat-header-cell *matHeaderCellDef>Место</th>
              <td mat-cell *matCellDef="let element">
                <span [class.medal-gold]="element.rank === 1" 
                      [class.medal-silver]="element.rank === 2"
                      [class.medal-bronze]="element.rank === 3">
                  {{ element.rank }}
                </span>
              </td>
            </ng-container>

            <ng-container matColumnDef="user">
              <th mat-header-cell *matHeaderCellDef>Пользователь</th>
              <td mat-cell *matCellDef="let element">{{ element.user }}</td>
            </ng-container>

            <ng-container matColumnDef="points">
              <th mat-header-cell *matHeaderCellDef>Очки</th>
              <td mat-cell *matCellDef="let element">
                <strong>{{ element.points }}</strong>
              </td>
            </ng-container>

            <tr mat-header-row *matHeaderRowDef="displayedColumns"></tr>
            <tr mat-row *matRowDef="let row; columns: displayedColumns;"></tr>
          </table>

          <div *ngIf="leaderboard.length === 0" class="empty-message">
            Нет данных для отображения
          </div>
        </mat-card-content>
      </mat-card>
    </div>
  `,
  styles: [`
    .leaderboard-container {
      max-width: 1200px;
      margin: 0 auto;
    }
    mat-card {
      margin-bottom: 20px;
    }
    .leaderboard-table {
      width: 100%;
    }
    .medal-gold {
      color: #FFD700;
      font-weight: bold;
      font-size: 1.2em;
    }
    .medal-silver {
      color: #C0C0C0;
      font-weight: bold;
      font-size: 1.2em;
    }
    .medal-bronze {
      color: #CD7F32;
      font-weight: bold;
      font-size: 1.2em;
    }
    .empty-message {
      padding: 20px;
      text-align: center;
      color: #666;
    }
  `]
})
export class LeaderboardComponent implements OnInit {
  leaderboard: any[] = [];
  displayedColumns: string[] = ['rank', 'user', 'points'];

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.loadLeaderboard();
  }

  loadLeaderboard() {
    this.apiService.getLeaderboard().subscribe({
      next: (leaderboard) => {
        this.leaderboard = leaderboard;
      },
      error: (err) => {
        console.error('Error loading leaderboard:', err);
        this.leaderboard = [];
      }
    });
  }
}

