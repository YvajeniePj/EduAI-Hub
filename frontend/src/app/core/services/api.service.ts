import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

const API_URL = '/api';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  constructor(private http: HttpClient) { }


  getUserById(id: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/users/${id}`);
  }

  getUserByName(name: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/users/by-name/${encodeURIComponent(name)}`);
  }

  createUser(name: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/users`, { name });
  }

  updateUser(id: string, userUpdate: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/users/${id}`, userUpdate);
  }

  uploadAvatar(id: string, formData: FormData): Observable<any> {
    return this.http.post<any>(`${API_URL}/users/${id}/avatar`, formData);
  }

  // Subjects
  getSubjects(): Observable<any[]> {
    return this.http.get<any[]>(`${API_URL}/subjects`);
  }

  createSubject(name: string, description?: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/subjects`, { name, description: description || null });
  }

  deleteSubject(id: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/subjects/${id}`);
  }

  uploadSubjectCover(subjectId: string, formData: FormData): Observable<any> {
    return this.http.post<any>(`${API_URL}/subjects/${subjectId}/cover`, formData);
  }

  // Tests
  getTests(subjectId?: string): Observable<any[]> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    return this.http.get<any[]>(`${API_URL}/tests`, { params });
  }

  getTest(id: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/tests/${id}`);
  }

  createTest(test: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/tests`, test);
  }

  updateTest(id: string, test: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/tests/${id}`, test);
  }

  deleteTest(id: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/tests/${id}`);
  }

  // Submissions
  getSubmissions(testId?: string, user?: string): Observable<any[]> {
    let params = new HttpParams();
    if (testId) {
      params = params.set('test_id', testId);
    }
    if (user) {
      params = params.set('user', user);
    }
    return this.http.get<any[]>(`${API_URL}/submissions`, { params });
  }

  createSubmission(submission: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/submissions`, submission);
  }

  updateSubmission(id: string, submission: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/submissions/${id}`, submission);
  }

  finishSubmission(id: string, useAi: boolean = false): Observable<any> {
    return this.http.post<any>(`${API_URL}/submissions/${id}/finish?use_ai=${useAi}`, {});
  }

  getSubmissionResults(id: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/submissions/${id}/results`);
  }

  // Materials
  getMaterials(subjectId?: string): Observable<any[]> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    return this.http.get<any[]>(`${API_URL}/materials`, { params });
  }

  uploadMaterial(formData: FormData): Observable<any> {
    return this.http.post<any>(`${API_URL}/materials`, formData);
  }

  createMaterialAnnotation(materialId: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/materials/${materialId}/annotate`, {});
  }

  deleteMaterial(materialId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/materials/${materialId}`);
  }

  // Videos
  getVideos(subjectId?: string): Observable<any[]> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    return this.http.get<any[]>(`${API_URL}/videos`, { params });
  }

  createVideo(video: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/videos`, video);
  }

  deleteVideo(videoId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/videos/${videoId}`);
  }

  // AI
  getAiStatus(): Observable<any> {
    return this.http.get<any>(`${API_URL}/ai/status`);
  }

  generateTest(request: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/ai/generate-test`, request);
  }

  chat(question: string, subjectId?: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/ai/chat`, { question, subject_id: subjectId });
  }

  getTestFeedback(feedbackRequest: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/ai/test-feedback`, feedbackRequest);
  }

  // Peer Review
  getSubmissionsForReview(testId: string, reviewer: string): Observable<any[]> {
    let params = new HttpParams();
    params = params.set('test_id', testId);
    params = params.set('reviewer', reviewer);
    return this.http.get<any[]>(`${API_URL}/reviews/submissions-for-review`, { params });
  }

  getMyReviews(user: string, testId?: string): Observable<any[]> {
    let params = new HttpParams();
    params = params.set('user', user);
    if (testId) {
      params = params.set('test_id', testId);
    }
    return this.http.get<any[]>(`${API_URL}/reviews/my-reviews`, { params });
  }

  createReview(review: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/reviews`, review);
  }

  getReviews(submissionId?: string): Observable<any[]> {
    let params = new HttpParams();
    if (submissionId) {
      params = params.set('submission_id', submissionId);
    }
    return this.http.get<any[]>(`${API_URL}/reviews`, { params });
  }

  // Gamification
  getLeaderboard(subjectId?: string, limit?: number): Observable<any[]> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    if (limit) {
      params = params.set('limit', limit.toString());
    }
    return this.http.get<any[]>(`${API_URL}/points`, { params });
  }

  exportLeaderboard(subjectId?: string): Observable<Blob> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    return this.http.get(`${API_URL}/points/export`, { params, responseType: 'blob' });
  }

  getUserPoints(username: string, subjectId?: string): Observable<any> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    return this.http.get<any>(`${API_URL}/points/${username}`, { params });
  }

  // News
  getNews(subjectId?: string): Observable<any[]> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    return this.http.get<any[]>(`${API_URL}/news`, { params });
  }

  createNews(news: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/news`, news);
  }

  updateNews(newsId: string, news: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/news/${newsId}`, news);
  }

  deleteNews(newsId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/news/${newsId}`);
  }

  // Groups
  getGroups(subjectId?: string, userName?: string): Observable<any[]> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    if (userName) {
      params = params.set('user_name', userName);
    }
    return this.http.get<any[]>(`${API_URL}/groups`, { params });
  }

  getGroup(groupId: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/groups/${groupId}`);
  }

  createGroup(group: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/groups`, group);
  }

  updateGroup(groupId: string, group: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/groups/${groupId}`, group);
  }

  deleteGroup(groupId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/groups/${groupId}`);
  }

  getGroupMembers(groupId: string): Observable<any[]> {
    return this.http.get<any[]>(`${API_URL}/groups/${groupId}/members`);
  }

  addGroupMember(groupId: string, member: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/groups/${groupId}/members`, member);
  }

  removeGroupMember(groupId: string, memberId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/groups/${groupId}/members/${memberId}`);
  }

  // Notifications
  getNotifications(userName?: string, isRead?: boolean): Observable<any[]> {
    let params = new HttpParams();
    if (userName) {
      params = params.set('user_name', userName);
    }
    if (isRead !== undefined) {
      params = params.set('is_read', isRead.toString());
    }
    return this.http.get<any[]>(`${API_URL}/notifications`, { params });
  }

  getNotificationCount(userName: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/notifications/count?user_name=${encodeURIComponent(userName)}`);
  }

  createNotification(notification: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/notifications`, notification);
  }

  markNotificationRead(notificationId: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/notifications/${notificationId}/mark-read`, {});
  }

  markAllNotificationsRead(userName: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/notifications/mark-all-read?user_name=${encodeURIComponent(userName)}`, {});
  }

  deleteNotification(notificationId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/notifications/${notificationId}`);
  }

  // Feedback
  getFeedbacks(userName?: string, subjectId?: string, groupId?: string): Observable<any[]> {
    let params = new HttpParams();
    if (userName) {
      params = params.set('user_name', userName);
    }
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    if (groupId) {
      params = params.set('group_id', groupId);
    }
    return this.http.get<any[]>(`${API_URL}/feedbacks`, { params });
  }

  getFeedbackStats(subjectId?: string, groupId?: string): Observable<any> {
    let params = new HttpParams();
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    if (groupId) {
      params = params.set('group_id', groupId);
    }
    return this.http.get<any>(`${API_URL}/feedbacks/stats`, { params });
  }

  createFeedback(feedback: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/feedbacks`, feedback);
  }

  deleteFeedback(feedbackId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/feedbacks/${feedbackId}`);
  }

  // Analytics
  createActivity(activity: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/analytics/activities`, activity);
  }

  getActivities(userName?: string, actionType?: string): Observable<any[]> {
    let params = new HttpParams();
    if (userName) {
      params = params.set('user_name', userName);
    }
    if (actionType) {
      params = params.set('action_type', actionType);
    }
    return this.http.get<any[]>(`${API_URL}/analytics/activities`, { params });
  }

  getProgress(userName?: string, subjectId?: string, groupId?: string): Observable<any[]> {
    let params = new HttpParams();
    if (userName) {
      params = params.set('user_name', userName);
    }
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    if (groupId) {
      params = params.set('group_id', groupId);
    }
    return this.http.get<any[]>(`${API_URL}/analytics/progress`, { params });
  }

  getAnalyticsReport(subjectId?: string, groupId?: string, userName?: string, days: number = 30): Observable<any> {
    let params = new HttpParams().set('days', days.toString());
    if (subjectId) {
      params = params.set('subject_id', subjectId);
    }
    if (groupId) {
      params = params.set('group_id', groupId);
    }
    if (userName) {
      params = params.set('user_name', userName);
    }
    return this.http.get<any>(`${API_URL}/analytics/report`, { params });
  }

  getActivityStats(userName: string, days: number = 30): Observable<any[]> {
    return this.http.get<any[]>(`${API_URL}/analytics/activity-stats?user_name=${encodeURIComponent(userName)}&days=${days}`);
  }

  // Users (for group management)
  getUsers(search?: string): Observable<any[]> {
    let params = new HttpParams();
    if (search) {
      params = params.set('search', search);
    }
    return this.http.get<any[]>(`${API_URL}/users`, { params });
  }

  // Group Requests
  createGroupRequest(groupId: string, userName: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/groups/${groupId}/requests`, { user_name: userName });
  }

  getGroupRequests(groupId: string, status?: string): Observable<any[]> {
    let params = new HttpParams();
    if (status) {
      params = params.set('status', status);
    }
    return this.http.get<any[]>(`${API_URL}/groups/${groupId}/requests`, { params });
  }

  updateGroupRequest(groupId: string, requestId: string, status: string, reviewedBy: string): Observable<any> {
    return this.http.put<any>(`${API_URL}/groups/${groupId}/requests/${requestId}`, {
      status: status,
      reviewed_by: reviewedBy
    });
  }

  getMyGroupRequests(userName: string, status?: string): Observable<any[]> {
    let params = new HttpParams().set('user_name', userName);
    if (status) {
      params = params.set('status', status);
    }
    return this.http.get<any[]>(`${API_URL}/groups/requests/my`, { params });
  }

  // Course Structure
  getCourseStructure(subjectId: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/subjects/${subjectId}/structure`);
  }

  createModule(subjectId: string, module: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/subjects/${subjectId}/modules`, module);
  }

  updateModule(moduleId: string, module: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/modules/${moduleId}`, module);
  }

  deleteModule(moduleId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/modules/${moduleId}`);
  }

  createLesson(moduleId: string, lesson: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/modules/${moduleId}/lessons`, lesson);
  }

  updateLesson(lessonId: string, lesson: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/lessons/${lessonId}`, lesson);
  }

  deleteLesson(lessonId: string): Observable<any> {
    return this.http.delete<any>(`${API_URL}/lessons/${lessonId}`);
  }

  createContent(lessonId: string, content: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/lessons/${lessonId}/content`, content);
  }

  updateContent(contentId: string, content: any): Observable<any> {
    return this.http.put<any>(`${API_URL}/content/${contentId}`, content);
  }

  getContent(lessonId: string): Observable<any> {
    return this.http.get<any>(`${API_URL}/lessons/${lessonId}/content`);
  }

  // Streaming
  createStreamingRoom(roomData: any): Observable<any> {
    return this.http.post<any>(`${API_URL}/streaming/rooms/create`, roomData);
  }

  generateStreamingToken(request: { room_name: string, identity: string, is_teacher: boolean }): Observable<any> {
    return this.http.post<any>(`${API_URL}/streaming/tokens/generate`, request);
  }

  getActiveStreamingRooms(): Observable<any[]> {
    return this.http.get<any[]>(`${API_URL}/streaming/rooms/active`);
  }

  endStreamingRoom(roomName: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/streaming/rooms/${roomName}/end`, {});
  }

  // AI Course Generation
  generateCourse(topic: string, additionalInfo?: string): Observable<any> {
    return this.http.post<any>(`${API_URL}/ai/generate-course`, {
      topic,
      additional_info: additionalInfo
    });
  }
}
