import { Component, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router, RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import {
    Room,
    RoomEvent,
    RemoteParticipant,
    RemoteTrack,
    RemoteTrackPublication,
    Track,
    createLocalVideoTrack,
    createLocalAudioTrack,
    LocalVideoTrack,
    LocalAudioTrack,
    LocalTrack,
    DataPacket_Kind,
    VideoPresets,
    ScreenSharePresets,
    ParticipantEvent
} from 'livekit-client';

interface ChatMessage {
    user: string;
    text: string;
    time: Date;
    isMe: boolean;
}

@Component({
    selector: 'app-stream',
    standalone: true,
    imports: [
        CommonModule,
        RouterModule,
        MatCardModule,
        MatButtonModule,
        MatIconModule,
        MatProgressSpinnerModule,
        MatSnackBarModule,
        MatFormFieldModule,
        MatInputModule,
        FormsModule
    ],
    template: `
    <div class="stream-container">
      <div class="main-layout">
        <div class="video-container">
          <div class="video-grid" #videoGrid>
            <!-- Overlay for loading -->
            <div *ngIf="loading" class="loading-overlay">
              <mat-spinner diameter="50"></mat-spinner>
              <p>Подключение к трансляции...</p>
            </div>
            
            <!-- Overlay for offline -->
            <div *ngIf="!loading && !isActive" class="offline-overlay">
              <mat-icon>videocam_off</mat-icon>
              <p>Трансляция не активна</p>
              <button mat-raised-button color="primary" (click)="startBroadcast()">
                Начать трансляцию
              </button>
            </div>

            <div class="video-grid-inner" [class.single-participant]="trackMap.size === 1" #videoGridInner>
              <!-- Video elements will be appended here dynamically -->
            </div>
          </div>

          <div class="controls-bar" *ngIf="isActive">
            <button mat-raised-button color="primary" *ngIf="!isBroadcasting" (click)="startBroadcasting()">
              <mat-icon>videocam</mat-icon>
              Выйти в эфир
            </button>
            
            <ng-container *ngIf="isBroadcasting">
              <button mat-fab [color]="isCameraOn ? 'primary' : 'warn'" (click)="toggleCamera()" title="Камера">
                <mat-icon>{{ isCameraOn ? 'videocam' : 'videocam_off' }}</mat-icon>
              </button>
              <button mat-fab [color]="isMicOn ? 'primary' : 'warn'" (click)="toggleMic()" title="Микрофон">
                <mat-icon>{{ isMicOn ? 'mic' : 'mic_off' }}</mat-icon>
              </button>
              <button mat-fab [color]="isScreenSharing ? 'accent' : 'default'" (click)="toggleScreenShare()" title="Демонстрация экрана">
                <mat-icon>{{ isScreenSharing ? 'stop_screen_share' : 'screen_share' }}</mat-icon>
              </button>
              <button mat-raised-button color="warn" (click)="stopBroadcasting()">
                Прекратить эфир
              </button>
            </ng-container>

            <button mat-button color="warn" (click)="endRoom()" class="end-room-btn">
              Завершить для всех
            </button>
          </div>
        </div>

        <div class="chat-container">
          <div class="chat-header">
            <h3>Чат трансляции</h3>
          </div>
          <div class="chat-messages" #chatScroll>
            <div *ngFor="let msg of messages" class="message" [class.is-me]="msg.isMe">
              <div class="msg-user">{{ msg.user }}</div>
              <div class="msg-text">{{ msg.text }}</div>
              <div class="msg-time">{{ msg.time | date:'HH:mm' }}</div>
            </div>
          </div>
          <div class="chat-input">
            <mat-form-field appearance="outline" class="full-width dark-field">
              <input matInput [(ngModel)]="newMessage" (keyup.enter)="sendMessage()" placeholder="Напишите сообщение...">
              <button mat-icon-button matSuffix (click)="sendMessage()" [disabled]="!newMessage.trim()">
                <mat-icon>send</mat-icon>
              </button>
            </mat-form-field>
          </div>
        </div>
      </div>
    </div>
  `,
    styles: [`
    .stream-container {
      height: calc(100vh - 100px);
      padding: 0;
      display: flex;
      flex-direction: column;
      background: #121212;
      color: white;
    }

    .main-layout {
      flex: 1;
      display: flex;
      overflow: hidden;
    }

    .video-container {
      flex: 3;
      position: relative;
      display: flex;
      flex-direction: column;
      background: #000;
    }

    .video-container {
      flex: 3;
      position: relative;
      display: flex;
      flex-direction: column;
      background: #000;
      overflow: hidden;
    }

    .video-grid {
      flex: 1;
      position: relative;
      background: #000;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }

    .video-grid-inner {
      width: 100%;
      height: 100%;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      grid-auto-rows: min-content;
      gap: 20px;
      padding: 24px;
      overflow-y: auto;
      align-content: start;
      flex: 1;
    }

    .video-grid-inner.single-participant {
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 0;
      grid-template-columns: none;
      grid-auto-rows: none;
    }

    .video-grid-inner.single-participant .participant-tile {
      max-width: 100%;
      max-height: 100%;
      width: 100%;
      height: 100%;
      aspect-ratio: auto;
      border: none;
      box-shadow: none;
      border-radius: 0;
    }

    .loading-overlay, .offline-overlay {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 10;
      background: rgba(0,0,0,0.8);
      grid-column: 1 / -1;
      height: 100%;
    }

    .offline-overlay mat-icon {
      font-size: 64px;
      width: 64px;
      height: 64px;
      margin-bottom: 16px;
      opacity: 0.5;
    }

    .controls-bar {
      height: 80px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 20px;
      background: rgba(0,0,0,0.7);
      backdrop-filter: blur(10px);
      border-top: 1px solid #333;
    }

    .chat-container {
      flex: 1;
      min-width: 300px;
      max-width: 400px;
      border-left: 1px solid #333;
      display: flex;
      flex-direction: column;
      background: #1a1a1a;
    }

    .chat-header {
      padding: 16px;
      border-bottom: 1px solid #333;
      background: #252525;
    }

    .chat-header h3 {
        margin: 0;
        font-size: 16px;
        color: #e0e0e0;
    }

    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .message {
        background: #2d2d2d;
        padding: 10px 14px;
        border-radius: 12px;
        align-self: flex-start;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .message.is-me {
        background: #1976d2;
        align-self: flex-end;
    }

    .msg-user {
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 4px;
        color: #64b5f6;
    }
    
    .message.is-me .msg-user {
        color: #bbdefb;
    }

    .msg-text {
        font-size: 14px;
        line-height: 1.4;
        word-wrap: break-word;
        color: #ffffff;
    }

    .msg-time {
        font-size: 10px;
        text-align: right;
        opacity: 0.6;
        margin-top: 6px;
        color: #aaa;
    }

    .chat-input {
        padding: 16px;
        border-top: 1px solid #333;
        background: #252525;
    }

    .full-width {
        width: 100%;
    }

    .dark-field ::ng-deep .mat-mdc-text-field-wrapper {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    .dark-field ::ng-deep .mdc-text-field--outlined:not(.mdc-text-field--disabled) .mdc-notched-outline__leading,
    .dark-field ::ng-deep .mdc-text-field--outlined:not(.mdc-text-field--disabled) .mdc-notched-outline__notch,
    .dark-field ::ng-deep .mdc-text-field--outlined:not(.mdc-text-field--disabled) .mdc-notched-outline__trailing {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    .dark-field input {
        color: white !important;
    }

    ::ng-deep .video-grid video {
        width: 100%;
        height: 100%;
        object-fit: contain;
        background: #000;
        border-radius: 4px;
    }

    .participant-tile {
        position: relative;
        width: 100%;
        aspect-ratio: 16 / 9;
        background: #111;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #444;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        transition: transform 0.2s ease;
    }
    
    .participant-tile:hover {
        border-color: #64b5f6;
    }

    .participant-tile .edu-participant-label {
        position: absolute;
        bottom: 16px;
        left: 16px;
        background: rgba(0,0,0,0.75);
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        color: #fff;
        z-index: 10;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.15);
        pointer-events: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    ::ng-deep .mat-mdc-form-field-subscript-wrapper {
        display: none;
    }
    
    @media (max-width: 1200px) {
        .video-grid {
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        }
    }
  `]
})
export class StreamComponent implements OnInit, OnDestroy, AfterViewInit {
    @ViewChild('videoGrid') videoGrid!: ElementRef;
    @ViewChild('videoGridInner') videoGridInner!: ElementRef;
    @ViewChild('chatScroll') chatScroll!: ElementRef;

    subjectId: string = '';
    roomName: string = '';
    isActive: boolean = false;
    isBroadcasting: boolean = false;
    loading: boolean = true;

    room?: Room;
    localVideoTrack?: LocalVideoTrack;
    localAudioTrack?: LocalAudioTrack;

    isCameraOn: boolean = true;
    isMicOn: boolean = true;
    isScreenSharing: boolean = false;

    messages: ChatMessage[] = [];
    newMessage: string = '';

    currentUser: any;

    constructor(
        private route: ActivatedRoute,
        private router: Router,
        private apiService: ApiService,
        private auth: AuthService,
        private snackBar: MatSnackBar
    ) { }

    ngOnInit() {
        this.subjectId = this.route.snapshot.params['id'];
        this.currentUser = this.auth.getCurrentUser() || { name: 'Гость', role: 'student' };

        this.checkRoom();
    }

    ngAfterViewInit() {
        this.scrollToBottom();
    }

    ngOnDestroy() {
        this.cleanup();
    }

    async checkRoom() {
        this.loading = true;
        try {
            const activeRooms = await this.apiService.getActiveStreamingRooms().toPromise();
            const room = (activeRooms || []).find((r: any) => r && r.subject_id === this.subjectId);

            if (room) {
                this.roomName = room.room_name;
                this.isActive = true;
                await this.connect();
            } else {
                this.isActive = false;
                this.loading = false;
            }
        } catch (err) {
            console.error('Error checking room:', err);
            this.snackBar.open('Ошибка при проверке трансляции', 'OK', { duration: 3000 });
            this.loading = false;
        }
    }

    async startBroadcast() {
        this.loading = true;
        try {
            const room = await this.apiService.createStreamingRoom({
                subject_id: this.subjectId,
                teacher_name: this.currentUser.name
            }).toPromise();

            this.roomName = room.room_name;
            this.isActive = true;
            await this.connect(true);
            this.isBroadcasting = true;
        } catch (err) {
            console.error('Error starting broadcast:', err);
            this.snackBar.open('Ошибка при запуске трансляции', 'OK', { duration: 3000 });
            this.loading = false;
        }
    }

    async connect(requestPublish: boolean = false) {
        try {
            const tokenData = await this.apiService.generateStreamingToken({
                room_name: this.roomName,
                identity: this.currentUser.name,
                is_teacher: requestPublish // Request publish permission
            }).toPromise();

            this.room = new Room({
                adaptiveStream: true,
                dynacast: true,
                videoCaptureDefaults: {
                    resolution: VideoPresets.h720.resolution,
                },
                publishDefaults: {
                    videoEncoding: VideoPresets.h720.encoding,
                    screenShareEncoding: ScreenSharePresets.h1080fps30.encoding,
                }
            });

            // Setup events
            this.room
                .on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
                    this.handleTrackSubscribed(track, publication, participant);
                })
                .on(RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
                    this.handleTrackUnsubscribed(track, publication, participant);
                })
                .on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
                    this.handleDataReceived(payload, participant);
                })
                .on(RoomEvent.Disconnected, () => {
                    this.isActive = false;
                    this.cleanup();
                });

            // Handle local track publishing for self-view
            this.room.localParticipant.on(ParticipantEvent.LocalTrackPublished, (publication) => {
                if (publication.track && publication.track.kind === Track.Kind.Video) {
                    this.addTrackToGrid(
                        publication.track as any,
                        this.currentUser.name,
                        publication.source === Track.Source.ScreenShare
                    );
                }
            });

            this.room.localParticipant.on(ParticipantEvent.LocalTrackUnpublished, (publication) => {
                const track = publication.track;
                if (track && track.sid) {
                    const elem = this.trackMap.get(track.sid);
                    if (elem) {
                        elem.remove();
                        this.trackMap.delete(track.sid);
                    }
                }
            });

            // Force TCP for better reliability in local docker on Windows
            // Must be set before connect()
            // @ts-ignore
            this.room.engine.client.forceTCP = true;

            await this.room.connect(tokenData.server_url, tokenData.token, {
                autoSubscribe: true,
            });

            if (requestPublish) {
                // Ensure we are connected before publishing
                if (this.room.state === 'connected') {
                    await this.publishTracks();
                } else {
                    console.warn('Room not connected yet, waiting...');
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    await this.publishTracks();
                }
            }

            this.loading = false;
        } catch (err) {
            console.error('Connection error:', err);
            this.snackBar.open('Ошибка подключения к LiveKit', 'OK', { duration: 3000 });
            this.loading = false;
        }
    }

    async startBroadcasting() {
        this.loading = true;
        try {
            await this.publishTracks();
            this.isBroadcasting = true;
        } catch (err) {
            console.error('Error starting broadcast:', err);
            this.snackBar.open('Не удалось выйти в эфир', 'OK', { duration: 3000 });
        } finally {
            this.loading = false;
        }
    }

    async stopBroadcasting() {
        this.loading = true;
        try {
            if (this.localVideoTrack && this.localVideoTrack.sid) {
                await this.room?.localParticipant.unpublishTrack(this.localVideoTrack);
                const elem = this.trackMap.get(this.localVideoTrack.sid);
                elem?.remove();
                this.trackMap.delete(this.localVideoTrack.sid);
                this.localVideoTrack.stop();
                this.localVideoTrack = undefined;
            }
            if (this.localAudioTrack && this.localAudioTrack.sid) {
                await this.room?.localParticipant.unpublishTrack(this.localAudioTrack);
                this.localAudioTrack.stop();
                this.localAudioTrack = undefined;
            }
            this.isBroadcasting = false;
        } catch (err) {
            console.error('Error stopping broadcast:', err);
        } finally {
            this.loading = false;
        }
    }

    trackMap = new Map<string, HTMLElement>();

    async publishTracks() {
        if (!this.room) return;

        try {
            this.localVideoTrack = await createLocalVideoTrack();
            this.localAudioTrack = await createLocalAudioTrack();

            await this.room.localParticipant.publishTrack(this.localVideoTrack);
            await this.room.localParticipant.publishTrack(this.localAudioTrack);

            // Note: We don't call addTrackToGrid here because LocalTrackPublished event will handle it
        } catch (err) {
            console.error('Error publishing tracks:', err);
            throw err;
        }
    }

    private addTrackToGrid(track: LocalTrack | RemoteTrack, identity: string, isScreenShare: boolean = false) {
        if (!track.sid || track.kind !== Track.Kind.Video) return;

        // Check if track is already added to avoid duplicates
        if (this.trackMap.has(track.sid)) {
            return;
        }

        const container = document.createElement('div');
        container.className = 'participant-tile';

        const videoElem = track.attach();
        container.appendChild(videoElem);

        const label = document.createElement('div');
        label.className = 'edu-participant-label';
        label.innerText = identity + (isScreenShare ? ' (Экран)' : '');
        container.appendChild(label);

        this.videoGridInner.nativeElement.appendChild(container);
        this.trackMap.set(track.sid, container);
    }

    handleTrackSubscribed(track: RemoteTrack, publication: RemoteTrackPublication, participant: RemoteParticipant) {
        if (track.kind === Track.Kind.Video) {
            this.addTrackToGrid(track, participant.identity, publication.source === Track.Source.ScreenShare);
        } else if (track.kind === Track.Kind.Audio) {
            const element = track.attach();
            document.body.appendChild(element); // Hidden audio element
        }
    }

    handleTrackUnsubscribed(track: RemoteTrack, publication: RemoteTrackPublication, participant: RemoteParticipant) {
        if (track.sid) {
            const elem = this.trackMap.get(track.sid);
            if (elem) {
                elem.remove();
                this.trackMap.delete(track.sid);
            }
        }
        track.detach();
    }

    handleDataReceived(payload: Uint8Array, participant?: RemoteParticipant) {
        const data = JSON.parse(new TextDecoder().decode(payload));
        if (data.type === 'chat') {
            this.messages.push({
                user: participant?.identity || 'Аноним',
                text: data.text,
                time: new Date(),
                isMe: false
            });
            this.scrollToBottom();
        }
    }

    sendMessage() {
        if (!this.newMessage.trim() || !this.room) return;

        const payload = {
            type: 'chat',
            text: this.newMessage
        };

        const encoder = new TextEncoder();
        this.room.localParticipant.publishData(
            encoder.encode(JSON.stringify(payload)),
            { reliable: true }
        );

        this.messages.push({
            user: 'Вы',
            text: this.newMessage,
            time: new Date(),
            isMe: true
        });

        this.newMessage = '';
        this.scrollToBottom();
    }

    toggleCamera() {
        if (this.room) {
            this.isCameraOn = !this.isCameraOn;
            this.room.localParticipant.setCameraEnabled(this.isCameraOn);
        }
    }

    toggleMic() {
        if (this.room) {
            this.isMicOn = !this.isMicOn;
            this.room.localParticipant.setMicrophoneEnabled(this.isMicOn);
        }
    }

    async toggleScreenShare() {
        if (!this.room) return;

        try {
            if (!this.isScreenSharing) {
                await this.room.localParticipant.setScreenShareEnabled(true);
                this.isScreenSharing = true;
            } else {
                await this.room.localParticipant.setScreenShareEnabled(false);
                this.isScreenSharing = false;
            }
        } catch (err) {
            console.error('Error toggling screen share:', err);
        }
    }

    async endRoom() {
        if (!confirm('Вы уверены, что хотите завершить трансляцию для всех участников?')) return;

        try {
            await this.apiService.endStreamingRoom(this.roomName).toPromise();
            await this.room?.disconnect();
            this.cleanup();
            this.isActive = false;
            this.snackBar.open('Трансляция завершена', 'OK', { duration: 3000 });
        } catch (err) {
            console.error('Error ending broadcast:', err);
        }
    }

    cleanup() {
        if (this.room) {
            this.room.disconnect();
            this.room = undefined;
        }
        if (this.localVideoTrack) {
            this.localVideoTrack.stop();
            this.localVideoTrack.detach();
            this.localVideoTrack = undefined;
        }
        if (this.localAudioTrack) {
            this.localAudioTrack.stop();
            this.localAudioTrack = undefined;
        }
        if (this.videoGridInner?.nativeElement) {
            this.videoGridInner.nativeElement.innerHTML = '';
        }
        this.trackMap.clear();
        this.isBroadcasting = false;
    }

    scrollToBottom() {
        if (this.chatScroll) {
            setTimeout(() => {
                this.chatScroll.nativeElement.scrollTop = this.chatScroll.nativeElement.scrollHeight;
            }, 100);
        }
    }
}
