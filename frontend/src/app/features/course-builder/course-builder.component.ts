import { Component, OnInit, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { FormsModule, ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatTreeModule } from '@angular/material/tree';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatDialog, MatDialogModule, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ApiService } from '../../core/services/api.service';
import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTreeNestedDataSource } from '@angular/material/tree';

interface TreeNode {
  id: string;
  title: string;
  type: 'module' | 'lesson';
  lessonType?: string;
  orderIndex: number;
  children?: TreeNode[];
  moduleId?: string;
  content?: any;
}

@Component({
  selector: 'app-course-builder',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatInputModule,
    MatFormFieldModule,
    MatSelectModule,
    MatTreeModule,
    MatExpansionModule,
    MatDialogModule,
    MatSnackBarModule,
    MatProgressSpinnerModule,
    MatCheckboxModule,
    MatChipsModule,
    MatTooltipModule
  ],
  template: `
    <div class="builder-container">
      <div class="builder-header">
        <h1>Конструктор курса: {{ courseName }}</h1>
        <button mat-raised-button color="primary" (click)="saveStructure()" [disabled]="saving">
          <mat-icon>save</mat-icon>
          Сохранить
        </button>
      </div>

      <div class="builder-content">
        <div class="sidebar">
          <mat-card class="structure-card">
            <mat-card-header>
              <mat-card-title>Структура курса</mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <button mat-raised-button color="primary" (click)="addModule()" class="add-button">
                <mat-icon>add</mat-icon>
                Добавить модуль
              </button>
              
              <mat-tree [dataSource]="dataSource" [treeControl]="treeControl" class="structure-tree">
                <!-- Узлы без детей (Lessons) -->
                <mat-tree-node *matTreeNodeDef="let node" matTreeNodePadding>
                  <button mat-icon-button (click)="selectNode(node)">
                    <mat-icon>{{ getNodeIcon(node.type) }}</mat-icon>
                  </button>
                  <span class="node-title">{{ node.title }}</span>
                  <!-- Кнопки для модулей, которые пока без детей -->
                  <ng-container *ngIf="node.type === 'module'">
                    <button mat-icon-button (click)="addSubModule(node)" class="add-btn" matTooltip="Добавить подмодуль">
                      <mat-icon>folder</mat-icon>
                    </button>
                    <button mat-icon-button (click)="addLesson(node)" class="add-btn" matTooltip="Добавить урок">
                      <mat-icon>add</mat-icon>
                    </button>
                  </ng-container>
                  <button mat-icon-button (click)="deleteNode(node)" color="warn" class="delete-btn">
                    <mat-icon>delete</mat-icon>
                  </button>
                </mat-tree-node>

                <!-- Узлы с детьми (Modules) -->
                <mat-nested-tree-node *matTreeNodeDef="let node; when: hasChild" matTreeNodePadding>
                  <div class="mat-tree-node">
                    <button mat-icon-button matTreeNodeToggle [attr.aria-label]="'Toggle ' + node.title">
                      <mat-icon class="mat-icon-rtl-mirror">
                        {{ treeControl.isExpanded(node) ? 'expand_more' : 'chevron_right' }}
                      </mat-icon>
                    </button>
                    <button mat-icon-button (click)="selectNode(node)">
                      <mat-icon>{{ getNodeIcon(node.type) }}</mat-icon>
                    </button>
                    <span class="node-title">{{ node.title }}</span>
                    <button mat-icon-button (click)="addSubModule(node)" class="add-btn" matTooltip="Добавить подмодуль">
                      <mat-icon>folder</mat-icon>
                    </button>
                    <button mat-icon-button (click)="addLesson(node)" class="add-btn" matTooltip="Добавить урок">
                      <mat-icon>add</mat-icon>
                    </button>
                    <button mat-icon-button (click)="deleteNode(node)" color="warn" class="delete-btn">
                      <mat-icon>delete</mat-icon>
                    </button>
                  </div>
                  <div [class.example-tree-invisible]="!treeControl.isExpanded(node)" role="group" class="nested-group">
                    <ng-container matTreeNodeOutlet></ng-container>
                  </div>
                </mat-nested-tree-node>
              </mat-tree>
            </mat-card-content>
          </mat-card>
        </div>

        <div class="content-area">
          <mat-card *ngIf="selectedNode" class="editor-card">
            <mat-card-header>
              <mat-card-title>
                {{ selectedNode.type === 'module' ? 'Редактирование модуля' : 'Редактирование урока' }}
              </mat-card-title>
            </mat-card-header>
            <mat-card-content>
              <form [formGroup]="editForm" (ngSubmit)="saveNode()">
                <mat-form-field appearance="outline" class="full-width">
                  <mat-label>Название</mat-label>
                  <input matInput formControlName="title" required>
                </mat-form-field>

                <mat-form-field *ngIf="selectedNode.type === 'lesson'" appearance="outline" class="full-width">
                  <mat-label>Тип урока</mat-label>
                  <mat-select formControlName="lessonType">
                    <mat-option value="lecture">Лекция</mat-option>
                    <mat-option value="quiz">Опрос</mat-option>
                    <mat-option value="video">Видео</mat-option>
                    <mat-option value="material">Материал</mat-option>
                    <mat-option value="exercise">Упражнение</mat-option>
                  </mat-select>
                </mat-form-field>

                <div *ngIf="selectedNode.type === 'lesson'" class="content-editor">
                  <h3>Контент урока</h3>
                  
                  <!-- Лекция - текстовый контент -->
                  <!-- Текстовый контент (для всех типов) -->
                  <div class="content-section">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Текст / Описание</mat-label>
                      <textarea matInput formControlName="textContent" rows="15" placeholder="Введите текст лекции или описание к уроку..."></textarea>
                    </mat-form-field>
                  </div>

                  <!-- Видео -->
                  <div *ngIf="editForm.get('lessonType')?.value === 'video'" class="content-section">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>URL видео (YouTube/Rutube)</mat-label>
                      <input matInput formControlName="videoUrl" placeholder="https://www.youtube.com/watch?v=... или https://rutube.ru/video/...">
                      <mat-hint>Вставьте ссылку на видео с YouTube или Rutube</mat-hint>
                    </mat-form-field>
                    <div *ngIf="editForm.get('videoUrl')?.value" class="video-preview">
                      <iframe 
                        [src]="getVideoEmbedUrl(editForm.get('videoUrl')?.value)" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen
                        style="width: 100%; height: 400px; border-radius: 8px;">
                      </iframe>
                    </div>
                  </div>

                  <!-- Материал -->
                  <div *ngIf="editForm.get('lessonType')?.value === 'material'" class="content-section">
                    <div class="material-selector">
                      <h4>Выберите материал:</h4>
                      <div class="material-actions">
                        <button mat-raised-button color="accent" (click)="loadMaterials()" type="button">
                          <mat-icon>refresh</mat-icon>
                          Обновить список
                        </button>
                        <button mat-raised-button color="primary" (click)="openMaterialUploadDialog()" type="button">
                          <mat-icon>cloud_upload</mat-icon>
                          Загрузить новый материал
                        </button>
                      </div>
                      <mat-form-field appearance="outline" class="full-width">
                        <mat-label>Материал</mat-label>
                        <mat-select formControlName="materialId">
                          <mat-option *ngFor="let material of availableMaterials" [value]="material.id">
                            {{ material.original_name || material.name }}
                            <span *ngIf="material.note"> - {{ material.note }}</span>
                          </mat-option>
                        </mat-select>
                      </mat-form-field>
                      <div *ngIf="editForm.get('materialId')?.value" class="selected-material">
                        <mat-icon>description</mat-icon>
                        <span>Выбран материал: {{ getMaterialName(editForm.get('materialId')?.value) }}</span>
                      </div>
                    </div>
                  </div>

                  <!-- Тест/Опрос -->
                  <div *ngIf="editForm.get('lessonType')?.value === 'quiz'" class="content-section">
                    <div class="test-selector">
                      <h4>Выберите или создайте тест:</h4>
                      <div class="test-actions">
                        <button mat-raised-button color="accent" (click)="loadTests()" type="button">
                          <mat-icon>refresh</mat-icon>
                          Обновить список
                        </button>
                        <button mat-raised-button color="primary" (click)="openCreateTestDialog()" type="button">
                          <mat-icon>add</mat-icon>
                          Создать тест
                        </button>
                        <button mat-raised-button color="primary" (click)="openGenerateTestDialog()" type="button">
                          <mat-icon>auto_awesome</mat-icon>
                          Сгенерировать через AI
                        </button>
                      </div>
                      <mat-form-field appearance="outline" class="full-width">
                        <mat-label>Тест</mat-label>
                        <mat-select formControlName="testId">
                          <mat-option *ngFor="let test of availableTests" [value]="test.id">
                            {{ test.title }}
                            <span *ngIf="test.description"> - {{ test.description }}</span>
                          </mat-option>
                        </mat-select>
                      </mat-form-field>
                      <div *ngIf="editForm.get('testId')?.value" class="selected-test">
                        <mat-icon>quiz</mat-icon>
                        <span>Выбран тест: {{ getTestName(editForm.get('testId')?.value) }}</span>
                        <button mat-icon-button (click)="viewTest(editForm.get('testId')?.value)" type="button">
                          <mat-icon>open_in_new</mat-icon>
                        </button>
                      </div>
                    </div>
                  </div>

                  <!-- Упражнение -->
                  <div *ngIf="editForm.get('lessonType')?.value === 'exercise'" class="content-section">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Описание упражнения</mat-label>
                      <textarea matInput formControlName="textContent" rows="10" placeholder="Опишите задание для упражнения..."></textarea>
                    </mat-form-field>
                  </div>
                </div>

                <!-- Редактирование модуля -->
                <div *ngIf="selectedNode.type === 'module'" class="module-editor">
                  <mat-form-field appearance="outline" class="full-width">
                    <mat-label>Описание модуля</mat-label>
                    <textarea matInput formControlName="description" rows="5" placeholder="Добавьте описание модуля (опционально)"></textarea>
                  </mat-form-field>
                  
                  <div class="module-actions">
                    <h4>Быстрые действия:</h4>
                    <div class="quick-actions">
                      <button mat-raised-button color="accent" (click)="addSubModule(selectedNode)" type="button">
                        <mat-icon>folder</mat-icon>
                        Добавить подмодуль
                      </button>
                      <button mat-raised-button color="primary" (click)="addLesson(selectedNode)" type="button">
                        <mat-icon>add</mat-icon>
                        Добавить урок
                      </button>
                    </div>
                  </div>
                </div>

                <div class="form-actions">
                  <button mat-raised-button color="primary" type="submit" [disabled]="!editForm.valid">
                    Сохранить
                  </button>
                  <button mat-button type="button" (click)="cancelEdit()">Отмена</button>
                </div>
              </form>
            </mat-card-content>
          </mat-card>

          <div *ngIf="!selectedNode" class="empty-state">
            <mat-icon>edit</mat-icon>
            <p>Выберите модуль или урок для редактирования</p>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .builder-container {
      min-height: 100vh;
      background: #f5f5f5;
      padding: 24px;
    }

    .builder-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
    }

    .builder-content {
      display: grid;
      grid-template-columns: 350px 1fr;
      gap: 24px;
    }

    .sidebar {
      position: sticky;
      top: 24px;
      height: fit-content;
    }

    .structure-card {
      margin-bottom: 24px;
    }

    .add-button {
      width: 100%;
      margin-bottom: 16px;
    }

    .structure-tree {
      background: transparent;
    }

    .node-title {
      flex: 1;
      margin-left: 8px;
    }

    .delete-btn, .add-btn {
      margin-left: auto;
    }

    .content-area {
      min-height: 600px;
    }

    .editor-card {
      margin-bottom: 24px;
    }

    .full-width {
      width: 100%;
      margin-bottom: 16px;
    }

    .content-editor {
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid #e0e0e0;
    }

    .form-actions {
      display: flex;
      gap: 16px;
      margin-top: 24px;
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
    }

    h1 {
      color: #1976d2;
      font-weight: 500;
      margin: 0;
    }

    .structure-card {
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border-radius: 8px;
    }

    .editor-card {
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      border-radius: 8px;
    }

    .node-title {
      font-weight: 500;
      color: #333;
    }

    .add-button {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      font-weight: 500;
      transition: transform 0.2s, box-shadow 0.2s;
    }

    .add-button:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    .structure-tree {
      max-height: 600px;
      overflow-y: auto;
    }

    .structure-tree::-webkit-scrollbar {
      width: 8px;
    }

    .structure-tree::-webkit-scrollbar-track {
      background: #f1f1f1;
      border-radius: 4px;
    }

    .structure-tree::-webkit-scrollbar-thumb {
      background: #888;
      border-radius: 4px;
    }

    .structure-tree::-webkit-scrollbar-thumb:hover {
      background: #555;
    }

    mat-tree-node {
      padding: 8px 0;
      border-radius: 4px;
      transition: background-color 0.2s;
    }

    mat-tree-node:hover {
      background-color: #f5f5f5;
    }

    .delete-btn {
      opacity: 0.6;
      transition: opacity 0.2s;
    }

    .delete-btn:hover {
      opacity: 1;
      color: #f44336;
    }

    .add-btn {
      opacity: 0.7;
      transition: opacity 0.2s, transform 0.2s;
    }

    .add-btn:hover {
      opacity: 1;
      transform: scale(1.1);
      color: #4caf50;
    }

    mat-card-header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 16px 24px;
      margin: -16px -16px 16px -16px;
      border-radius: 8px 8px 0 0;
    }

    mat-card-title {
      color: white;
      font-weight: 500;
    }

    .builder-header button {
      box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .form-actions button[type="submit"] {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
    }

    .content-editor {
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid #e0e0e0;
    }

    .content-section {
      margin-bottom: 24px;
    }

    .content-section h4 {
      margin: 0 0 16px 0;
      color: #333;
      font-weight: 500;
    }

    .material-actions, .test-actions {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }

    .selected-material, .selected-test {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px;
      background: #f5f5f5;
      border-radius: 8px;
      margin-top: 12px;
    }

    .selected-material mat-icon, .selected-test mat-icon {
      color: #667eea;
    }

    .video-preview {
      margin-top: 16px;
      border-radius: 8px;
      overflow: hidden;
    }

    .module-editor {
      margin-top: 16px;
    }

    .module-actions {
      margin-top: 24px;
      padding-top: 24px;
      border-top: 1px solid #e0e0e0;
    }

    .module-actions h4 {
      margin: 0 0 16px 0;
      color: #333;
      font-weight: 500;
    }

    .quick-actions {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .quick-actions button {
      flex: 1;
      min-width: 150px;
    }

    .example-tree-invisible {
      display: none;
    }
    
    .nested-group {
      padding-left: 24px;
    }

    .mat-tree-node {
      display: flex;
      align-items: center; 
      min-height: 48px;
    }
  `]
})
export class CourseBuilderComponent implements OnInit {
  subjectId: string = '';
  courseName: string = '';
  structure: any = null;
  selectedNode: TreeNode | null = null;
  editForm: FormGroup;
  saving = false;
  availableMaterials: any[] = [];
  availableTests: any[] = [];
  uploadingMaterial = false;

  dataSource = new MatTreeNestedDataSource<TreeNode>();
  treeControl = new NestedTreeControl<TreeNode>(node => node.children);

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private apiService: ApiService,
    private fb: FormBuilder,
    private snackBar: MatSnackBar,
    private dialog: MatDialog
  ) {
    this.editForm = this.fb.group({
      title: ['', Validators.required],
      description: [''],
      lessonType: ['lecture'],
      textContent: [''],
      videoUrl: [''],
      materialId: [''],
      testId: ['']
    });
  }

  ngOnInit() {
    this.subjectId = this.route.snapshot.params['id'];
    this.loadCourse();
    this.loadMaterials();
    this.loadTests();
  }

  loadCourse() {
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
      },
      error: (err) => {
        console.error('Error loading course structure:', err);
        this.snackBar.open('Ошибка загрузки структуры курса', 'Закрыть');
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

    this.dataSource.data = nodes;
  }

  hasChild = (_: number, node: TreeNode) => !!node.children && node.children.length > 0;

  toggleNode(node: TreeNode) {
    if (this.treeControl.isExpanded(node)) {
      this.treeControl.collapse(node);
    } else {
      this.treeControl.expand(node);
    }
  }

  getNodeIcon(type: string): string {
    const icons: { [key: string]: string } = {
      'module': 'folder',
      'lecture': 'article',
      'quiz': 'quiz',
      'video': 'video_library',
      'material': 'description',
      'exercise': 'fitness_center'
    };
    return icons[type] || 'circle';
  }

  selectNode(node: TreeNode) {
    this.selectedNode = node;
    this.editForm.patchValue({
      title: node.title,
      description: node.type === 'module' ? (this.structure?.modules?.find((m: any) => m.id === node.id)?.description || '') : '',
      lessonType: node.lessonType || 'lecture',
      textContent: node.content?.text_content || '',
      videoUrl: node.content?.video_url || '',
      materialId: node.content?.material_id || '',
      testId: node.content?.test_id || ''
    });
  }

  addModule() {
    const dialogRef = this.dialog.open(InputDialogComponent, {
      width: '400px',
      data: { title: 'Добавить модуль', label: 'Название модуля:', placeholder: 'Введите название модуля' }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.trim()) {
        this.apiService.createModule(this.subjectId, {
          title: result.trim(),
          description: '',
          is_collapsed: false
        }).subscribe({
          next: () => {
            this.loadCourse();
            this.snackBar.open('Модуль добавлен', 'Закрыть', { duration: 2000 });
          },
          error: (err) => {
            console.error('Error creating module:', err);
            this.snackBar.open('Ошибка создания модуля: ' + (err.error?.detail || err.message || 'Неизвестная ошибка'), 'Закрыть', { duration: 5000 });
          }
        });
      }
    });
  }

  addSubModule(parentModule: TreeNode) {
    const dialogRef = this.dialog.open(InputDialogComponent, {
      width: '400px',
      data: { title: 'Добавить подмодуль', label: 'Название подмодуля:', placeholder: 'Введите название подмодуля' }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.trim()) {
        // Создаем новый модуль как подмодуль (в будущем можно добавить parent_module_id)
        this.apiService.createModule(this.subjectId, {
          title: result.trim(),
          description: `Подмодуль модуля "${parentModule.title}"`,
          is_collapsed: false
        }).subscribe({
          next: () => {
            this.loadCourse();
            this.snackBar.open('Подмодуль добавлен', 'Закрыть', { duration: 2000 });
          },
          error: (err) => {
            console.error('Error creating submodule:', err);
            this.snackBar.open('Ошибка создания подмодуля: ' + (err.error?.detail || err.message || 'Неизвестная ошибка'), 'Закрыть', { duration: 5000 });
          }
        });
      }
    });
  }

  addLesson(moduleNode: TreeNode) {
    const dialogRef = this.dialog.open(InputDialogComponent, {
      width: '400px',
      data: { title: 'Добавить урок', label: 'Название урока:', placeholder: 'Введите название урока' }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result && result.trim()) {
        this.apiService.createLesson(moduleNode.id, {
          title: result.trim(),
          lesson_type: 'lecture'
        }).subscribe({
          next: () => {
            this.loadCourse();
            this.snackBar.open('Урок добавлен', 'Закрыть', { duration: 2000 });
          },
          error: (err) => {
            console.error('Error creating lesson:', err);
            this.snackBar.open('Ошибка создания урока: ' + (err.error?.detail || err.message || 'Неизвестная ошибка'), 'Закрыть', { duration: 5000 });
          }
        });
      }
    });
  }

  deleteNode(node: TreeNode) {
    if (!confirm(`Удалить "${node.title}" ? `)) return;

    if (node.type === 'module') {
      this.apiService.deleteModule(node.id).subscribe({
        next: () => {
          this.loadCourse();
          this.selectedNode = null;
          this.snackBar.open('Модуль удален', 'Закрыть', { duration: 2000 });
        }
      });
    } else {
      this.apiService.deleteLesson(node.id).subscribe({
        next: () => {
          this.loadCourse();
          this.selectedNode = null;
          this.snackBar.open('Урок удален', 'Закрыть', { duration: 2000 });
        }
      });
    }
  }

  saveNode() {
    if (!this.selectedNode || !this.editForm.valid) return;

    const formValue = this.editForm.value;

    if (this.selectedNode.type === 'module') {
      this.apiService.updateModule(this.selectedNode.id, {
        title: formValue.title,
        description: formValue.description || null
      }).subscribe({
        next: () => {
          this.loadCourse();
          this.snackBar.open('Модуль сохранен', 'Закрыть', { duration: 2000 });
        }
      });
    } else {
      // Update lesson
      this.apiService.updateLesson(this.selectedNode.id, {
        title: formValue.title,
        lesson_type: formValue.lessonType
      }).subscribe({
        next: () => {
          // Update content
          const contentData: any = {};
          contentData.lesson_id = this.selectedNode!.id; // Add lesson_id for validation
          if (formValue.textContent) contentData.text_content = formValue.textContent;
          if (formValue.videoUrl) {
            contentData.video_url = formValue.videoUrl;
            contentData.video_platform = formValue.videoUrl.includes('youtube') ? 'youtube' : 'rutube';
          }
          if (formValue.materialId) contentData.material_id = formValue.materialId;
          if (formValue.testId) contentData.test_id = formValue.testId;

          if (this.selectedNode?.content?.id) {
            this.apiService.updateContent(this.selectedNode.content.id, contentData).subscribe({
              next: () => {
                this.loadCourse();
                this.snackBar.open('Урок сохранен', 'Закрыть', { duration: 2000 });
              }
            });
          } else {
            this.apiService.createContent(this.selectedNode!.id, contentData).subscribe({
              next: () => {
                this.loadCourse();
                this.snackBar.open('Урок сохранен', 'Закрыть', { duration: 2000 });
              }
            });
          }
        }
      });
    }
  }

  cancelEdit() {
    this.selectedNode = null;
    this.editForm.reset();
  }

  saveStructure() {
    this.saving = true;
    // Structure is saved automatically when editing nodes
    setTimeout(() => {
      this.saving = false;
      this.snackBar.open('Структура сохранена', 'Закрыть', { duration: 2000 });
    }, 500);
  }

  loadMaterials() {
    if (!this.subjectId) return;
    this.apiService.getMaterials(this.subjectId).subscribe({
      next: (materials) => {
        this.availableMaterials = materials;
      },
      error: (err) => {
        console.error('Error loading materials:', err);
      }
    });
  }

  loadTests() {
    if (!this.subjectId) return;
    this.apiService.getTests(this.subjectId).subscribe({
      next: (tests) => {
        this.availableTests = tests;
      },
      error: (err) => {
        console.error('Error loading tests:', err);
      }
    });
  }

  getMaterialName(materialId: string): string {
    const material = this.availableMaterials.find(m => m.id === materialId);
    return material ? (material.original_name || material.name) : '';
  }

  getTestName(testId: string): string {
    const test = this.availableTests.find(t => t.id === testId);
    return test ? test.title : '';
  }

  getVideoEmbedUrl(url: string): string {
    if (!url) return '';
    // YouTube
    if (url.includes('youtube.com/watch?v=')) {
      const videoId = url.split('v=')[1]?.split('&')[0];
      return `https://www.youtube.com/embed/${videoId}`;
    }
    if (url.includes('youtu.be/')) {
      const videoId = url.split('youtu.be/')[1]?.split('?')[0];
      return `https://www.youtube.com/embed/${videoId}`;
    }
    // Rutube
    if (url.includes('rutube.ru/video/')) {
      const videoId = url.split('rutube.ru/video/')[1]?.split('/')[0];
      return `https://rutube.ru/play/embed/${videoId}`;
    }
    return '';
  }

  openMaterialUploadDialog() {
    const dialogRef = this.dialog.open(MaterialUploadDialogComponent, {
      width: '500px',
      data: { subjectId: this.subjectId }
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.loadMaterials();
        this.editForm.patchValue({ materialId: result.id });
        this.snackBar.open('Материал загружен и выбран', 'Закрыть', { duration: 2000 });
      }
    });
  }

  openCreateTestDialog() {
    this.router.navigate(['/tests'], { queryParams: { subjectId: this.subjectId, returnTo: 'builder' } });
  }

  openGenerateTestDialog() {
    this.router.navigate(['/ai-test'], { queryParams: { subjectId: this.subjectId, returnTo: 'builder' } });
  }

  viewTest(testId: string) {
    this.router.navigate(['/tests', testId]);
  }
}

@Component({
  selector: 'app-input-dialog',
  standalone: true,
  imports: [CommonModule, FormsModule, MatDialogModule, MatFormFieldModule, MatInputModule, MatButtonModule],
  template: `
    <h2 mat-dialog-title>{{ data.title }}</h2>
    <mat-dialog-content>
      <mat-form-field appearance="outline" class="full-width">
        <mat-label>{{ data.label }}</mat-label>
        <input matInput [(ngModel)]="inputValue" [placeholder]="data.placeholder" (keyup.enter)="onSubmit()" autofocus>
      </mat-form-field>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="onCancel()">Отмена</button>
      <button mat-raised-button color="primary" (click)="onSubmit()" [disabled]="!inputValue || !inputValue.trim()">OK</button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width {
      width: 100%;
      min-width: 300px;
    }
    mat-dialog-content {
      padding: 20px 24px;
    }
    mat-dialog-actions {
      padding: 8px 24px 16px;
    }
  `]
})
export class InputDialogComponent {
  inputValue: string = '';

  constructor(
    public dialogRef: MatDialogRef<InputDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { title: string; label: string; placeholder: string }
  ) { }

  onCancel(): void {
    this.dialogRef.close();
  }

  onSubmit(): void {
    if (this.inputValue?.trim()) {
      this.dialogRef.close(this.inputValue);
    }
  }
}

@Component({
  selector: 'app-material-upload-dialog',
  standalone: true,
  imports: [CommonModule, FormsModule, MatDialogModule, MatFormFieldModule, MatInputModule, MatButtonModule, MatProgressSpinnerModule, MatIconModule],
  template: `
    <h2 mat-dialog-title>Загрузить материал</h2>
    <mat-dialog-content>
      <input type="file" #fileInput (change)="onFileSelected($event)" style="display: none" accept=".pdf,.doc,.docx,.txt">
      <button mat-raised-button (click)="fileInput.click()" [disabled]="uploading">
        <mat-icon>attach_file</mat-icon>
        Выбрать файл
      </button>
      <div *ngIf="selectedFile" style="margin-top: 16px;">
        <p><strong>Выбран файл:</strong> {{ selectedFile.name }}</p>
        <p><strong>Размер:</strong> {{ formatSize(selectedFile.size) }}</p>
      </div>
      <mat-form-field appearance="outline" class="full-width" style="margin-top: 16px;">
        <mat-label>Описание (опционально)</mat-label>
        <input matInput [(ngModel)]="note" placeholder="Например: лекция 1, слайды">
      </mat-form-field>
      <mat-spinner *ngIf="uploading" diameter="30" style="margin: 20px auto;"></mat-spinner>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="onCancel()" [disabled]="uploading">Отмена</button>
      <button mat-raised-button color="primary" (click)="onUpload()" [disabled]="!selectedFile || uploading">
        {{ uploading ? 'Загрузка...' : 'Загрузить' }}
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .full-width {
      width: 100%;
      min-width: 300px;
    }
    mat-dialog-content {
      padding: 20px 24px;
    }
    mat-dialog-actions {
      padding: 8px 24px 16px;
    }
  `]
})
export class MaterialUploadDialogComponent {
  selectedFile: File | null = null;
  note: string = '';
  uploading = false;

  constructor(
    public dialogRef: MatDialogRef<MaterialUploadDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { subjectId: string },
    private apiService: ApiService,
    private snackBar: MatSnackBar
  ) { }

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
  }

  formatSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  onCancel(): void {
    this.dialogRef.close();
  }

  onUpload(): void {
    if (!this.selectedFile) return;

    this.uploading = true;
    const formData = new FormData();
    formData.append('file', this.selectedFile);
    formData.append('subject_id', this.data.subjectId);
    if (this.note) {
      formData.append('note', this.note);
    }

    this.apiService.uploadMaterial(formData).subscribe({
      next: (material) => {
        this.uploading = false;
        this.dialogRef.close(material);
        this.snackBar.open('Материал успешно загружен', 'Закрыть', { duration: 2000 });
      },
      error: (err) => {
        this.uploading = false;
        this.snackBar.open('Ошибка загрузки: ' + (err.error?.detail || err.message || 'Неизвестная ошибка'), 'Закрыть', { duration: 5000 });
      }
    });
  }
}
