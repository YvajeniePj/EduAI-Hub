import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule, Router } from '@angular/router';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTreeModule, MatTreeNestedDataSource } from '@angular/material/tree';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTableModule } from '@angular/material/table';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { MatSelectModule } from '@angular/material/select';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { CreateGroupDialogComponent } from '../groups/groups.component';
import { UploadMaterialDialogComponent } from './upload-material-dialog.component';

interface TreeNode {
  id: string;
  title: string;
  type: 'module' | 'lesson';
  lessonType?: string;
  orderIndex: number;
  children?: TreeNode[];
  moduleId?: string;
  content?: any;
  isCollapsed?: boolean;
}

@Component({
  selector: 'app-course-view',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatTreeModule,
    MatExpansionModule,
    MatProgressSpinnerModule,
    MatTabsModule,
    MatTableModule,
    MatDialogModule,
    MatSelectModule,
    FormsModule,
    ReactiveFormsModule,
    CreateGroupDialogComponent,
    UploadMaterialDialogComponent
  ],
  template: `
    <div class="course-hub-container">
      <div class="course-header" *ngIf="!loading">
        <h1>{{ courseName }}</h1>
        <div class="header-actions">
             <!-- Actions like Edit Course could go here -->
        </div>
      </div>

      <mat-tab-group animationDuration="0ms" class="course-tabs" [selectedIndex]="0">
        <!-- Tab 1: Training (Existing View) -->
        <mat-tab label="Обучение">
          <div class="course-layout">
      <!-- Sidebar -->
      <div class="sidebar">
        <div class="sidebar-header">
          <h2>Содержание</h2>
        </div>
        <div class="sidebar-content">
          <mat-tree [dataSource]="dataSource" [treeControl]="treeControl" class="nav-tree">
            <!-- Lesson Node (Leaf) -->
            <mat-tree-node *matTreeNodeDef="let node" matTreeNodePadding>
              <button mat-button class="nav-item-btn" [class.active]="selectedLesson?.id === node.id" (click)="selectLesson(node)">
                <span class="tree-indicator"></span>
                <span class="nav-text">{{ node.title }}</span>
              </button>
            </mat-tree-node>

            <!-- Module Node (Parent) -->
            <mat-nested-tree-node *matTreeNodeDef="let node; when: hasChild" matTreeNodePadding>
              <div class="module-group">
                <button mat-icon-button matTreeNodeToggle [attr.aria-label]="'Toggle ' + node.title">
                  <mat-icon class="mat-icon-rtl-mirror">
                    {{ treeControl.isExpanded(node) ? 'expand_more' : 'chevron_right' }}
                  </mat-icon>
                </button>
                <span class="module-title">{{ node.title }}</span>
              </div>
              <div [class.example-tree-invisible]="!treeControl.isExpanded(node)" role="group">
                <ng-container matTreeNodeOutlet></ng-container>
              </div>
            </mat-nested-tree-node>
          </mat-tree>
        </div>
      </div>

      <!-- Main Content -->
      <div class="main-content">
        <!-- Breadcrumbs -->
        <div class="breadcrumbs" *ngIf="selectedLesson">
           <span>{{ courseName }}</span>
           <mat-icon class="separator">chevron_right</mat-icon>
           <span>{{ getModuleName(selectedLesson.moduleId) }}</span>
           <mat-icon class="separator">chevron_right</mat-icon>
           <span class="current">{{ selectedLesson.title }}</span>
        </div>

        <div *ngIf="loading" class="loading-container">
          <mat-spinner diameter="40"></mat-spinner>
        </div>

        <div *ngIf="!loading && selectedLesson" class="content-area">
          <div class="lesson-header">
            <div class="header-icon">
               <mat-icon>menu_book</mat-icon>
            </div>
            <div class="header-text">
               <h1>Учебник</h1>
               <div class="lesson-title">{{ selectedLesson.title }}</div>
            </div>
            <span class="spacer"></span>
            <button mat-raised-button color="warn" *ngIf="isStreamActive" [routerLink]="['/courses', subjectId, 'stream']" class="live-btn">
              <mat-icon>videocam</mat-icon>
              В ЭФИРЕ
            </button>
          </div>

          <div class="content-body">
              <!-- Text content -->
              <div *ngIf="selectedLesson.content?.text_content" class="text-content">
                <div [innerHTML]="selectedLesson.content.text_content"></div>
              </div>

              <!-- Video content -->
              <div *ngIf="selectedLesson.content?.video_url" class="video-section">
                <h3>Видеоматериал</h3>
                <div class="video-container">
                    <iframe
                    [src]="getVideoEmbedUrl(selectedLesson.content.video_url)"
                    frameborder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen
                    class="video-iframe">
                    </iframe>
                </div>
              </div>

               <!-- Material link -->
              <ng-container *ngIf="selectedLesson.content?.material_id">
                  <div *ngIf="isContentAllowed(selectedLesson.content.material_id, 'material')" class="resource-card">
                    <mat-icon class="resource-icon">description</mat-icon>
                    <div class="resource-info">
                       <div class="resource-title">Материал для скачивания</div>
                       <button mat-button color="primary" (click)="downloadMaterial(selectedLesson.content.material_id)">
                         Скачать файл
                       </button>
                    </div>
                  </div>
                  <div *ngIf="!isContentAllowed(selectedLesson.content.material_id, 'material')" class="resource-card locked">
                      <mat-icon class="resource-icon">lock</mat-icon>
                      <div class="resource-info">
                          <div class="resource-title">Материал недоступен для вашей группы</div>
                      </div>
                  </div>
              </ng-container>

              <!-- Test link -->
              <ng-container *ngIf="selectedLesson.content?.test_id">
                  <div *ngIf="isContentAllowed(selectedLesson.content.test_id, 'test')" class="resource-card test-card">
                    <mat-icon class="resource-icon">quiz</mat-icon>
                    <div class="resource-info">
                       <div class="resource-title">Проверочное тестирование</div>
                       <button mat-raised-button color="primary" [routerLink]="['/tests', selectedLesson.content.test_id, 'take']">
                         Начать тест
                       </button>
                    </div>
                  </div>
                  <div *ngIf="!isContentAllowed(selectedLesson.content.test_id, 'test')" class="resource-card locked">
                      <mat-icon class="resource-icon">lock</mat-icon>
                      <div class="resource-info">
                          <div class="resource-title">Тест недоступен для вашей группы</div>
                      </div>
                  </div>
              </ng-container>

              <div *ngIf="!selectedLesson.content || (!selectedLesson.content.text_content && !selectedLesson.content.video_url && !selectedLesson.content.material_id && !selectedLesson.content.test_id)" class="empty-content">
                <p>Содержимое урока пока не добавлено.</p>
              </div>
          </div>
        </div>

        <div *ngIf="!loading && !selectedLesson" class="select-hint">
          <mat-icon>touch_app</mat-icon>
          <p>Выберите главу из содержания, чтобы начать обучение</p>
        </div>
      </div>
    </div>
        </mat-tab>

        <!-- Tab 2: Assignments (Teacher Only) -->
        <mat-tab label="Задания и Материалы" *ngIf="currentUser?.role === 'teacher'">
            <div class="tab-content-container">
                <div class="section-header">
                    <h2>Управление контентом</h2>
                    <div class="actions">
                        <button mat-raised-button color="primary" (click)="openCreateTest()">
                            <mat-icon>quiz</mat-icon> Создать тест
                        </button>
                        <button mat-raised-button color="accent" (click)="openUploadMaterial()">
                            <mat-icon>upload_file</mat-icon> Загрузить материал
                        </button>
                    </div>
                </div>
                
                <div class="assignments-list">
                    <div class="list-section">
                        <h3>Тесты</h3>
                        <div *ngIf="visibleTests.length === 0" class="empty-list">Нет тестов</div>
                        <mat-card *ngFor="let test of visibleTests" class="item-card">
                            <mat-card-content class="item-content">
                                <div class="item-info">
                                    <div class="item-title-row">
                                        <mat-icon class="item-icon">quiz</mat-icon>
                                        <span class="item-title">{{ test.title }}</span>
                                    </div>
                                    <div class="item-meta">
                                        <span class="meta-label">Доступ:</span> {{ getGroupNames(test.allowed_groups) }}
                                        <span *ngIf="test.due_date" class="meta-separator">•</span>
                                        <span *ngIf="test.due_date">Дедлайн: {{ test.due_date | date:'short' }}</span>
                                    </div>
                                </div>
                                <div class="item-actions">
                                    <button mat-icon-button color="warn" (click)="deleteTest(test.id)">
                                        <mat-icon>delete</mat-icon>
                                    </button>
                                </div>
                            </mat-card-content>
                        </mat-card>
                    </div>

                    <div class="list-section">
                        <h3>Материалы</h3>
                        <div *ngIf="visibleMaterials.length === 0" class="empty-list">Нет материалов</div>
                        <mat-card *ngFor="let material of visibleMaterials" class="item-card">
                             <mat-card-content class="item-content">
                                <div class="item-info">
                                    <div class="item-title-row">
                                        <mat-icon class="item-icon">description</mat-icon>
                                        <span class="item-title">{{ material.original_name || material.name }}</span>
                                    </div>
                                    <div class="item-meta">
                                        <span class="meta-label">Доступ:</span> {{ getGroupNames(material.allowed_groups) }}
                                        <span class="meta-separator">•</span>
                                        <span>{{ material.note || 'Без описания' }}</span>
                                    </div>
                                </div>
                                <div class="item-actions">
                                    <button mat-button color="primary" (click)="downloadMaterial(material.id)">Скачать</button>
                                    <button mat-icon-button color="warn" (click)="deleteMaterial(material.id)">
                                        <mat-icon>delete</mat-icon>
                                    </button>
                                </div>
                            </mat-card-content>
                        </mat-card>
                    </div>
                </div>
            </div>
        </mat-tab>

        <!-- Tab 3: Groups (Teacher & Student) -->
        <mat-tab label="Группы">
             <div class="tab-content-container">
                <div class="section-header">
                    <h2>Группы курса</h2>
                    <button mat-raised-button color="primary" (click)="createGroup()" *ngIf="currentUser?.role === 'teacher'">
                        <mat-icon>group_add</mat-icon> Создать группу
                    </button>
                </div>

                <!-- Teacher Action -->
                <div class="list-section" *ngIf="currentUser?.role === 'teacher'">
                    <p class="section-hint">Управляйте участниками и заявками на странице каждой группы.</p>
                </div>
                
                <div class="groups-list">
                    <div *ngIf="groups.length === 0" class="empty-list">Нет доступных групп</div>
                    <mat-card *ngFor="let group of groups" class="item-card">
                         <mat-card-content class="item-content">
                            <div class="item-info">
                                <div class="item-title-row">
                                    <mat-icon class="item-icon">group</mat-icon>
                                    <span class="item-title">{{ group.name }}</span>
                                </div>
                                <div class="item-meta">
                                    <span>Участников: {{ group.member_count || 0 }}</span>
                                    <span class="meta-separator">•</span>
                                    <span>{{ group.description || 'Без описания' }}</span>
                                </div>
                            </div>
                            <div class="item-actions">
                                <!-- Teacher actions -->
                                <div *ngIf="currentUser?.role === 'teacher'" class="teacher-btns">
                                    <button mat-icon-button color="primary" (click)="navigateToGroup(group.id)" matTooltip="Управление">
                                        <mat-icon>settings</mat-icon>
                                    </button>
                                    <button mat-icon-button color="warn" (click)="deleteGroup(group.id)">
                                        <mat-icon>delete</mat-icon>
                                    </button>
                                </div>
                                
                                <!-- Student actions -->
                                <div *ngIf="currentUser?.role === 'student'">
                                    <span *ngIf="isGroupMember(group.id)" class="status-badge member">Вы участник</span>
                                    <span *ngIf="!isGroupMember(group.id) && hasPendingRequest(group.id)" class="status-badge pending">Заявка отправлена</span>
                                    <button mat-raised-button color="primary" *ngIf="!isGroupMember(group.id) && !hasPendingRequest(group.id)" (click)="joinGroup(group.id)">
                                        Вступить
                                    </button>
                                </div>
                            </div>
                         </mat-card-content>
                    </mat-card>
                </div>
            </div>
        </mat-tab>

      </mat-tab-group>
    </div>
  `,
  styles: [`
    .course-hub-container {
      display: flex;
      flex-direction: column;
      height: 100vh;
      background-color: #f5f5f5;
    }

    .course-header {
      background: white;
      padding: 16px 24px;
      border-bottom: 1px solid #e0e0e0;
      flex-shrink: 0;
    }

    .course-header h1 {
        margin: 0;
        font-size: 24px;
        color: #1a237e;
    }

    /* Tabs Override */
    ::ng-deep .course-tabs .mat-mdc-tab-body-wrapper {
        flex: 1; 
        height: 100%;
    }
    
    ::ng-deep .course-tabs {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .tab-content-container {
        padding: 24px;
        height: 100%;
        overflow-y: auto;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }
    
    .actions {
        display: flex;
        gap: 12px;
    }
    
    .info-card {
        padding: 24px;
        text-align: center;
        color: #666;
    }
    
    .list-section {
        margin-bottom: 32px;
    }
    
    .list-section h3 {
        margin: 0 0 16px 0;
        font-size: 18px;
        color: #1a237e;
        border-bottom: 2px solid #e8eaf6;
        padding-bottom: 8px;
    }
    
    .item-card {
        margin-bottom: 12px;
        border-left: 4px solid #3f51b5;
    }
    
    .item-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px !important;
    }
    
    .item-info {
        flex: 1;
    }
    
    .item-title-row {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    
    .item-icon {
        margin-right: 8px;
        color: #3f51b5;
    }
    
    .item-title {
        font-weight: 500;
        font-size: 16px;
    }
    
    .item-meta {
        font-size: 13px;
        color: #666;
        margin-left: 32px; 
    }
    
    .meta-label {
        font-weight: 500;
        color: #333;
    }
    
    .meta-separator {
        margin: 0 8px;
        color: #ccc;
    }
    
    .empty-list {
        padding: 24px;
        text-align: center;
        color: #999;
        font-style: italic;
        background: white;
        border-radius: 4px;
    }

    .info-card-bg {
        background-color: #f0f7ff !important;
        border-left: 4px solid #1976d2 !important;
    }

    .status-badge {
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 500;
    }

    .status-badge.member {
        background-color: #e8f5e9;
        color: #2e7d32;
    }

    .status-badge.pending {
        background-color: #fff3e0;
        color: #e65100;
    }

    .teacher-btns {
        display: flex;
        gap: 8px;
    }

    .section-hint {
        color: #666;
        font-style: italic;
        margin-bottom: 16px;
    }


    .course-hub-container {
      display: flex;
      flex-direction: column;
      height: 100vh;
      background-color: #f5f5f5;
    }

    .course-header {
      background: white;
      padding: 16px 24px;
      border-bottom: 1px solid #e0e0e0;
      flex-shrink: 0;
    }

    .course-header h1 {
        margin: 0;
        font-size: 24px;
        color: #1a237e;
    }

    /* Tabs Override */
    ::ng-deep .course-tabs .mat-mdc-tab-body-wrapper {
        flex: 1; 
        height: 100%;
    }
    
    ::ng-deep .course-tabs {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .course-layout {
      display: flex;
      height: 100%; /* Fill the tab body */
      background-color: #f5f5f5;
      overflow: hidden;
    }
    
    .tab-content-container {
        padding: 24px;
        height: 100%;
        overflow-y: auto;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }
    
    .actions {
        display: flex;
        gap: 12px;
    }

    /* Sidebar */
    .sidebar {
      width: 320px;
      background: white;
      border-right: 1px solid #e0e0e0;
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }

    .sidebar-header {
      padding: 20px 24px;
      border-bottom: 1px solid #f0f0f0;
    }

    .sidebar-header h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
      color: #000;
    }

    .sidebar-content {
      flex: 1;
      overflow-y: auto;
      padding: 12px 0;
    }

    .nav-tree {
      background: transparent;
    }

    .module-group {
      display: flex;
      align-items: center;
      padding: 4px 8px;
      font-weight: 500;
      color: #333;
    }

    .module-title {
        font-size: 14px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .nav-item-btn {
      width: 100%;
      text-align: left;
      padding: 8px 16px 8px 48px; /* Indent for lessons */
      font-size: 14px;
      color: #555;
      position: relative;
      display: flex;
      align-items: center;
      justify-content: flex-start;
      border-radius: 0;
    }
    
    .nav-item-btn:hover {
        background-color: #f5f5f5;
    }

    .nav-item-btn.active {
      background-color: #e3f2fd;
      color: #1565c0;
      font-weight: 500;
    }
    
    .nav-item-btn.active::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background-color: #1565c0;
    }
    
    .nav-text {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Main Content */
    .main-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: #fafafa;
    }

    .breadcrumbs {
      padding: 16px 32px;
      display: flex;
      align-items: center;
      font-size: 13px;
      color: #757575;
      border-bottom: 1px solid #e0e0e0;
      background: white;
    }

    .separator {
      font-size: 16px;
      width: 16px;
      height: 16px;
      margin: 0 8px;
      color: #bdbdbd;
    }

    .current {
      color: #1565c0;
      font-weight: 500;
    }

    .content-area {
      flex: 1;
      padding: 32px 48px;
      overflow-y: auto;
      background: white;
      max-width: 1000px; /* Readability limit */
      width: 100%;
      margin: 0 auto;
      box-shadow: 0 0 10px rgba(0,0,0,0.02);
    }

    .lesson-header {
      display: flex;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 32px;
      border-bottom: 1px solid #eee;
      padding-bottom: 24px;
    }

    .header-icon {
        width: 48px;
        height: 48px;
        background-color: #e8f5e9;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #2e7d32;
    }

    .header-icon mat-icon {
        font-size: 28px;
        width: 28px;
        height: 28px;
    }

    .header-text h1 {
        margin: 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #757575;
        font-weight: 500;
    }

    .lesson-title {
        font-size: 24px;
        font-weight: 600;
        color: #212121;
        margin-top: 4px;
    }

    .text-content {
      font-size: 16px;
      line-height: 1.6;
      color: #212121;
      margin-bottom: 32px;
    }

    .video-section {
        margin: 32px 0;
    }
    
    .video-section h3 {
        margin-bottom: 16px;
        font-size: 18px;
    }

    .video-container {
        position: relative;
        padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
        height: 0;
        overflow: hidden;
        border-radius: 8px;
        background: black;
    }
    
    .video-iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }

    .resource-card {
      display: flex;
      align-items: center;
      padding: 16px;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      margin-bottom: 16px;
      background: #fafafa;
    }

    .resource-icon {
        font-size: 32px;
        width: 32px;
        height: 32px;
        color: #1976d2;
        margin-right: 16px;
    }

    .resource-info {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .resource-title {
        font-weight: 500;
        color: #424242;
    }

    .test-card .resource-icon {
        color: #7b1fa2;
    }

    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
    }

    .select-hint {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #9e9e9e;
    }
    
    .select-hint mat-icon {
        font-size: 64px;
        width: 64px;
        height: 64px;
        margin-bottom: 16px;
        opacity: 0.5;
    }

    .live-btn {
        animation: pulse 2s infinite;
        font-weight: bold;
    }

    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }

    .spacer {
        flex: 1 1 auto;
    }

    .example-tree-invisible {
      display: none;
    }
  `]
})
export class CourseViewComponent implements OnInit {
  subjectId: string = '';
  courseName: string = '';
  structure: any = null;
  selectedLesson: TreeNode | null = null;
  loading = false;
  isStreamActive = false;
  currentUser: any;

  // Data for tabs
  materials: any[] = [];
  tests: any[] = [];
  groups: any[] = []; // All groups (Teacher/Student View)
  myGroups: any[] = []; // Current user's groups (Student)
  myRequests: any[] = []; // Requests made by student

  dataSource = new MatTreeNestedDataSource<TreeNode>();
  treeControl = new NestedTreeControl<TreeNode>(node => node.children);

  constructor(
    private route: ActivatedRoute,
    private apiService: ApiService,
    private auth: AuthService,
    private sanitizer: DomSanitizer,
    private router: Router,
    private dialog: MatDialog
  ) { }

  ngOnInit() {
    this.subjectId = this.route.snapshot.params['id'];
    this.currentUser = this.auth.getCurrentUser();
    this.loadCourse();
    this.checkActiveStream();

    // Load content for access control checks
    this.loadMaterials();
    this.loadTests();

    // Load management data
    this.loadGroups();
    if (this.currentUser?.role === 'student') {
      // Load my groups for access checking
      this.loadMyGroups();
      this.loadMyRequests();
    }
  }

  checkActiveStream() {
    this.apiService.getActiveStreamingRooms().subscribe({
      next: (rooms) => {
        this.isStreamActive = (rooms || []).some((r: any) => r && r.subject_id === this.subjectId);
      }
    });
  }

  loadCourse() {
    this.loading = true;

    this.apiService.getSubjects().subscribe({
      next: (subjects) => {
        const subject = subjects.find((s: any) => s.id === this.subjectId);
        if (subject) {
          this.courseName = subject.name;
        }
      }
    });

    this.apiService.getCourseStructure(this.subjectId).subscribe({
      next: (structure) => {
        this.structure = structure;
        this.buildTree();
        this.loading = false;

        // Auto-select first lesson if available
        if (this.dataSource.data.length > 0) {
          const firstModule = this.dataSource.data[0];
          if (firstModule.children && firstModule.children.length > 0) {
            this.selectLesson(firstModule.children[0]);
          }
        }
      },
      error: (err) => {
        console.error('Error loading course structure:', err);
        this.loading = false;
      }
    });
  }

  buildTree() {
    if (!this.structure || !this.structure.modules) return;

    const nodes: TreeNode[] = this.structure.modules.map((module: any) => ({
      id: module.id,
      title: module.title,
      type: 'module',
      orderIndex: module.order_index,
      isCollapsed: module.is_collapsed,
      children: (module.lessons || []).map((lesson: any) => ({
        id: lesson.id,
        title: lesson.title,
        type: 'lesson' as const,
        lessonType: lesson.lesson_type,
        orderIndex: lesson.order_index,
        moduleId: module.id,
        content: lesson.content
      }))
    }));

    // Sort accordingly if needed, but backend should handle order
    this.dataSource.data = nodes;

    // Expand all modules by default
    nodes.forEach(node => {
      if (node.children && node.children.length > 0 && !node.isCollapsed) {
        this.treeControl.expand(node);
      }
    });
  }

  hasChild = (_: number, node: TreeNode) => !!node.children && node.children.length > 0;

  selectLesson(lesson: TreeNode) {
    if (lesson.type !== 'lesson') return;
    this.selectedLesson = lesson;

    // Track video view if lesson has video
    if (lesson.content?.video_url) {
      const currentUser = this.auth.getCurrentUser();
      if (currentUser) {
        this.apiService.createActivity({
          user_name: currentUser.name,
          action_type: 'video_view',
          resource_type: 'video',
          resource_id: lesson.id // Using lesson id as resource id for simplicity
        }).subscribe();
      }
    }
  }

  getModuleName(moduleId?: string): string {
    if (!moduleId || !this.structure?.modules) return '';
    const module = this.structure.modules.find((m: any) => m.id === moduleId);
    return module ? module.title : '';
  }

  getVideoEmbedUrl(url: string): SafeResourceUrl {
    let embedUrl = '';

    if (url.includes('youtube.com/watch')) {
      const videoId = url.split('v=')[1]?.split('&')[0];
      embedUrl = `https://www.youtube.com/embed/${videoId}`;
    } else if (url.includes('youtu.be/')) {
      const videoId = url.split('youtu.be/')[1]?.split('?')[0];
      embedUrl = `https://www.youtube.com/embed/${videoId}`;
    } else if (url.includes('rutube.ru/video/')) {
      const videoId = url.split('rutube.ru/video/')[1]?.split('/')[0];
      embedUrl = `https://rutube.ru/play/embed/${videoId}`;
    } else {
      embedUrl = url;
    }

    return this.sanitizer.bypassSecurityTrustResourceUrl(embedUrl);
  }

  downloadMaterial(materialId: string) {
    // Track material view activity
    const currentUser = this.auth.getCurrentUser();
    if (currentUser) {
      this.apiService.createActivity({
        user_name: currentUser.name,
        action_type: 'material_view',
        resource_type: 'material',
        resource_id: materialId
      }).subscribe();
    }

    // Direct link to download endpoint - open in new tab for viewing
    const url = `/api/materials/${materialId}/download`;
    window.open(url, '_blank');
  }

  loadMaterials() {
    this.apiService.getMaterials(this.subjectId).subscribe({
      next: (data) => this.materials = data,
      error: (err) => console.error('Error loading materials', err)
    });
  }

  loadTests() {
    // Pass subjectId to filter tests
    this.apiService.getTests(this.subjectId).subscribe({
      next: (data) => this.tests = data,
      error: (err) => console.error('Error loading tests', err)
    });
  }

  loadGroups() {
    this.apiService.getGroups(this.subjectId).subscribe({
      next: (data) => this.groups = data,
      error: (err) => console.error('Error loading groups', err)
    });
  }

  loadMyGroups() {
    if (!this.currentUser) return;
    this.apiService.getGroups(this.subjectId, this.currentUser.name).subscribe({
      next: (data) => this.myGroups = data,
      error: (err) => console.error('Error loading my groups', err)
    });
  }

  isContentAllowed(contentId: string, type: 'test' | 'material'): boolean {
    if (this.currentUser?.role === 'teacher') return true; // Teachers see everything

    let item: any;
    if (type === 'test') {
      item = this.tests.find(t => t.id === contentId);
    } else {
      item = this.materials.find(m => m.id === contentId);
    }

    if (!item) {
      // If item not found in the preloaded lists, we can't determine groups yet.
      // We should probably hide it to be safe, but let's log it.
      console.warn(`[AccessControl] Item ${contentId} (${type}) not found in loaded lists.`);
      return false;
    }

    // If no groups are specified, it's public
    if (!item.allowed_groups || item.allowed_groups.length === 0) {
      return true;
    }

    // Log for debugging
    console.log(`[AccessControl] Checking ${type} "${item.title || item.name}":`, {
      allowed: item.allowed_groups,
      userGroups: this.myGroups.map(g => g.id),
      userName: this.currentUser?.name
    });

    // Check intersection: does student have ANY group that is in allowed_groups?
    const hasAccess = this.myGroups.some(g => item.allowed_groups.includes(g.id));

    if (!hasAccess) {
      console.warn(`[AccessControl] Access DENIED for ${type} "${item.title || item.name}"`);
    }

    return hasAccess;
  }

  loadMyRequests() {
    if (!this.currentUser) return;
    this.apiService.getMyGroupRequests(this.currentUser.name, 'pending').subscribe({
      next: (requests) => this.myRequests = requests,
      error: (err) => console.error('Error loading my requests', err)
    });
  }

  getGroupName(groupId: string): string {
    const g = this.groups.find(group => group.id === groupId);
    return g ? g.name : 'Unknown';
  }

  getGroupNames(groupIds: string[]): string {
    if (!groupIds || groupIds.length === 0) return 'Все';
    if (!this.groups || this.groups.length === 0) return '...';

    return groupIds.map(id => {
      const g = this.groups.find(group => group.id === id);
      return g ? g.name : 'Unknown';
    }).join(', ');
  }

  isGroupMember(groupId: string): boolean {
    return this.myGroups.some(g => g.id === groupId);
  }

  hasPendingRequest(groupId: string): boolean {
    return this.myRequests.some(r => r.group_id === groupId);
  }

  joinGroup(groupId: string) {
    if (!this.currentUser) return;
    this.apiService.createGroupRequest(groupId, this.currentUser.name).subscribe({
      next: () => {
        alert('Заявка отправлена!');
        this.loadMyRequests();
      },
      error: (err) => alert('Ошибка при отправке заявки: ' + (err.error?.detail || err.message))
    });
  }

  navigateToGroup(id: string) {
    this.router.navigate(['/groups', id]);
  }

  get visibleTests() {
    return this.tests.filter(t => this.isContentAllowed(t.id, 'test'));
  }

  get visibleMaterials() {
    return this.materials.filter(m => this.isContentAllowed(m.id, 'material'));
  }

  deleteTest(testId: string) {
    if (confirm('Удалить тест?')) {
      this.apiService.deleteTest(testId).subscribe({
        next: () => this.loadTests(),
        error: (err) => alert('Ошибка при удалении')
      });
    }
  }

  deleteMaterial(materialId: string) {
    if (confirm('Удалить материал?')) {
      this.apiService.deleteMaterial(materialId).subscribe({
        next: () => this.loadMaterials(),
        error: (err) => alert('Ошибка при удалении')
      });
    }
  }

  deleteGroup(groupId: string) {
    if (confirm('Удалить группу?')) {
      this.apiService.deleteGroup(groupId).subscribe({
        next: () => this.loadGroups(),
        error: (err) => alert('Ошибка при удалении')
      });
    }
  }

  // Placeholder methods for new tabs
  openCreateTest() {
    this.router.navigate(['/tests/create'], { queryParams: { subjectId: this.subjectId } });
  }

  openUploadMaterial() {
    const dialogRef = this.dialog.open(UploadMaterialDialogComponent, {
      width: '600px',
      data: { subjectId: this.subjectId }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.loadMaterials();
      }
    });
  }

  createGroup() {
    const dialogRef = this.dialog.open(CreateGroupDialogComponent, {
      width: '600px',
      data: {
        subjects: [{ id: this.subjectId, name: this.courseName }], // Restrict to current subject
        currentUser: this.currentUser
      }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        // Force subject_id to current subject just in case
        result.subject_id = this.subjectId;

        this.apiService.createGroup(result).subscribe({
          next: () => {
            this.loadGroups();
            alert('Группа создана!');
          },
          error: (err) => {
            console.error('Error creating group:', err);
            alert('Ошибка при создании группы: ' + (err.error?.detail || err.message));
          }
        });
      }
    });
  }
}
