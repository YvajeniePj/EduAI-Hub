import { Routes } from '@angular/router';
import { SubjectsComponent } from './features/subjects/subjects.component';
import { TestsComponent } from './features/tests/tests.component';
import { TestDetailComponent } from './features/tests/test-detail.component';
import { TestCreateComponent } from './features/tests/test-create.component';
import { TestTakeComponent } from './features/tests/test-take.component';
import { SubmissionsComponent } from './features/submissions/submissions.component';
import { SubmissionResultsComponent } from './features/submissions/submission-results.component';
import { MaterialsComponent } from './features/materials/materials.component';
import { LoginComponent } from './features/auth/login.component';
import { RegisterComponent } from './features/auth/register.component';
import { ProfileComponent } from './features/profile/profile.component';
import { AdminDbComponent } from './features/admin/admin-db.component';
import { HomeComponent } from './features/home/home.component';
import { NewsManageComponent } from './features/news/news-manage.component';
import { AnalyticsComponent } from './features/analytics/analytics.component';
import { ActivityMonitorComponent } from './features/activity-monitor/activity-monitor.component';
import { FeedbackResultsComponent } from './features/feedback-results/feedback-results.component';
import { GroupsComponent } from './features/groups/groups.component';
import { GroupDetailComponent } from './features/groups/group-detail.component';
import { StudentsComponent } from './features/students/students.component';
import { CourseBuilderComponent } from './features/course-builder/course-builder.component';
import { CourseBuilderListComponent } from './features/course-builder/course-builder-list.component';
import { CourseViewComponent } from './features/course-view/course-view.component';
import { authGuard } from './core/guards/auth.guard';

export const routes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  { path: '', component: HomeComponent, canActivate: [authGuard] },
  { path: 'news/manage', component: NewsManageComponent, canActivate: [authGuard] },
  { path: 'subjects', component: SubjectsComponent, canActivate: [authGuard] },
  { path: 'course-builder', component: CourseBuilderListComponent, canActivate: [authGuard] },
  { path: 'course-builder/:id', component: CourseBuilderComponent, canActivate: [authGuard] },
  { path: 'courses/:id', component: CourseViewComponent, canActivate: [authGuard] },
  { path: 'courses/:id/stream', loadComponent: () => import('./features/streaming/stream.component').then(m => m.StreamComponent), canActivate: [authGuard] },
  { path: 'streaming', loadComponent: () => import('./features/streaming/stream-list.component').then(m => m.StreamListComponent), canActivate: [authGuard] },
  { path: 'tests', component: TestsComponent, canActivate: [authGuard] },
  { path: 'tests/create', component: TestCreateComponent, canActivate: [authGuard] },
  { path: 'tests/:id', component: TestDetailComponent, canActivate: [authGuard] },
  { path: 'tests/:id/take', component: TestTakeComponent, canActivate: [authGuard] },
  { path: 'submissions', component: SubmissionsComponent, canActivate: [authGuard] },
  { path: 'submissions/:id/results', component: SubmissionResultsComponent, canActivate: [authGuard] },
  { path: 'materials', component: MaterialsComponent, canActivate: [authGuard] },
  { path: 'profile', component: ProfileComponent, canActivate: [authGuard] },
  { path: 'admin/db', component: AdminDbComponent, canActivate: [authGuard] },
  { path: 'analytics', component: AnalyticsComponent, canActivate: [authGuard] },
  { path: 'activity-monitor', component: ActivityMonitorComponent, canActivate: [authGuard] },
  { path: 'feedback-results', component: FeedbackResultsComponent, canActivate: [authGuard] },
  { path: 'groups', component: GroupsComponent, canActivate: [authGuard] },
  { path: 'groups/:id', component: GroupDetailComponent, canActivate: [authGuard] },
  { path: 'students', component: StudentsComponent, canActivate: [authGuard] },
  { path: 'videos', loadComponent: () => import('./features/videos/videos.component').then(m => m.VideosComponent), canActivate: [authGuard] },
  { path: 'peer-review', loadComponent: () => import('./features/peer-review/peer-review.component').then(m => m.PeerReviewComponent), canActivate: [authGuard] },
  { path: 'leaderboard', loadComponent: () => import('./features/leaderboard/leaderboard.component').then(m => m.LeaderboardComponent), canActivate: [authGuard] },
  { path: 'ai-test', loadComponent: () => import('./features/ai-test/ai-test.component').then(m => m.AiTestComponent), canActivate: [authGuard] },
  { path: 'chat', loadComponent: () => import('./features/chat/chat.component').then(m => m.ChatComponent), canActivate: [authGuard] }
];

