import cv2
import mediapipe as mp
import numpy as np
import time
from enum import Enum
from collections import deque
import sys
import json
import os
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class ExerciseType(Enum):
    SQUAT = 1
    PUSHUP = 2
    SHOULDER_PRESS = 3
    DIPS = 4
    BENCH_PRESS = 5
    PULL_UPS = 6

class RepQuality(Enum):
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, cap, form_corrector):
        super().__init__()
        self.cap = cap
        self.form_corrector = form_corrector
        self.running = True
        self.paused = False
        
    def run(self):
        while self.running and self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Process frame with form corrector
                frame, stats = self.form_corrector.process_frame(frame)
                
                self.frame_ready.emit(frame)
                self.stats_updated.emit(stats)
                
            self.msleep(33)  # ~30 FPS
            
    def pause(self):
        self.paused = True
        
    def resume(self):
        self.paused = False
        
    def stop(self):
        self.running = False

class WorkoutSession:
    def __init__(self):
        self.start_time = datetime.now()
        self.exercise_history = []
        self.total_reps = 0
        self.total_accuracy = 0
        self.best_rep_accuracy = 0
        self.worst_rep_accuracy = 100
        self.rep_qualities = []
        
    def add_rep(self, accuracy, quality):
        self.total_reps += 1
        self.rep_qualities.append(quality)
        self.total_accuracy = ((self.total_accuracy * (self.total_reps - 1)) + accuracy) / self.total_reps
        self.best_rep_accuracy = max(self.best_rep_accuracy, accuracy)
        self.worst_rep_accuracy = min(self.worst_rep_accuracy, accuracy)
        
    def get_quality_distribution(self):
        if not self.rep_qualities:
            return {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        
        distribution = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        for quality in self.rep_qualities:
            if isinstance(quality, RepQuality):
                distribution[quality.value] += 1
            else:
                distribution[quality] += 1
        return distribution

class AIFormCorrectorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.form_corrector = AIFormCorrector()
        self.video_thread = None
        self.cap = None
        self.workout_session = WorkoutSession()
        
        # Settings
        self.settings = {
            'confidence_threshold': 0.7,
            'smoothing_factor': 0.8,
            'rep_hold_frames': 5,
            'auto_save_sessions': True
        }
        
        self.init_ui()
        self.setStyleSheet(self.get_dark_theme())
        
        # Load previous settings
        self.load_settings()
        
    def init_ui(self):
        self.setWindowTitle("🏋️‍♂️ AI Form Corrector - Professional Edition v2.0")
        self.setGeometry(100, 100, 1600, 1000)
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        
        # Left panel - Controls
        self.create_left_panel(main_layout)
        
        # Center panel - Video feed
        self.create_center_panel(main_layout)
        
        # Right panel - Advanced stats
        self.create_right_panel(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready to start - Select video source")
        
        # Menu bar
        self.create_menu_bar()
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        save_session_action = QAction('Save Session', self)
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        load_session_action = QAction('Load Session', self)
        load_session_action.triggered.connect(self.load_session)
        file_menu.addAction(load_session_action)
        
        export_data_action = QAction('Export Data', self)
        export_data_action.triggered.connect(self.export_data)
        file_menu.addAction(export_data_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        fullscreen_action = QAction('Toggle Fullscreen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
    def create_left_panel(self, main_layout):
        left_panel = QFrame()
        left_panel.setFixedWidth(400)
        left_panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(left_panel)
        
        # Header
        header = QLabel("🎯 AI FORM CORRECTOR")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50; margin: 10px;")
        layout.addWidget(header)
        
        # Video source section
        self.create_video_source_section(layout)
        
        # Exercise selection
        self.create_exercise_section(layout)
        
        # Current rep info
        self.create_current_rep_section(layout)
        
        # Controls
        self.create_controls_section(layout)
        
        layout.addStretch()
        main_layout.addWidget(left_panel)
        
    def create_current_rep_section(self, layout):
        group = QGroupBox("🎯 Current Rep Analysis")
        group_layout = QVBoxLayout(group)
        
        # Rep counter with quality
        self.rep_display = QLabel("Reps: 0")
        self.rep_display.setAlignment(Qt.AlignCenter)
        self.rep_display.setStyleSheet("font-size: 28px; font-weight: bold; color: #2196F3;")
        group_layout.addWidget(self.rep_display)
        
        # Current rep accuracy
        self.current_accuracy_label = QLabel("Current Rep: 0%")
        self.current_accuracy_label.setAlignment(Qt.AlignCenter)
        self.current_accuracy_label.setStyleSheet("font-size: 18px; color: #FF9800;")
        group_layout.addWidget(self.current_accuracy_label)
        
        # Rep quality indicator
        self.quality_label = QLabel("Quality: Ready")
        self.quality_label.setAlignment(Qt.AlignCenter)
        self.quality_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        group_layout.addWidget(self.quality_label)
        
        # Movement phase
        self.phase_label = QLabel("Phase: Ready")
        self.phase_label.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(self.phase_label)
        
        # Range of motion indicator
        self.rom_progress = QProgressBar()
        self.rom_progress.setMaximum(100)
        self.rom_progress.setValue(0)
        self.rom_progress.setFormat("Range of Motion: %p%")
        group_layout.addWidget(self.rom_progress)
        
        # Real-time feedback
        self.live_feedback = QTextEdit()
        self.live_feedback.setMaximumHeight(120)
        self.live_feedback.setPlaceholderText("Live form feedback...")
        self.live_feedback.setReadOnly(True)
        group_layout.addWidget(self.live_feedback)
        
        layout.addWidget(group)
        
    def create_center_panel(self, main_layout):
        center_panel = QFrame()
        center_panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(center_panel)
        
        # Video display
        self.video_label = QLabel("Select a video source to begin")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel { 
                background-color: #1a1a1a; 
                border: 2px dashed #555;
                color: #888;
                font-size: 18px;
            }
        """)
        layout.addWidget(self.video_label)
        
        # Exercise info overlay
        self.exercise_info = QLabel("")
        self.exercise_info.setAlignment(Qt.AlignCenter)
        self.exercise_info.setStyleSheet("font-size: 16px; color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.exercise_info)
        
        # Video controls (for video files)
        self.video_controls = QFrame()
        controls_layout = QHBoxLayout(self.video_controls)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        controls_layout.addWidget(self.progress_slider)
        
        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)
        
        layout.addWidget(self.video_controls)
        self.video_controls.hide()
        
        main_layout.addWidget(center_panel)
        
    def create_right_panel(self, main_layout):
        right_panel = QFrame()
        right_panel.setFixedWidth(350)
        right_panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(right_panel)
        
        # Session statistics
        self.create_session_stats_section(layout)
        
        # Rep quality breakdown
        self.create_quality_breakdown_section(layout)
        
        # Performance trends
        self.create_performance_section(layout)
        
        layout.addStretch()
        main_layout.addWidget(right_panel)
        
    def create_session_stats_section(self, layout):
        group = QGroupBox("📊 Session Statistics")
        group_layout = QVBoxLayout(group)
        
        # Session duration
        self.session_time_label = QLabel("Duration: 00:00")
        group_layout.addWidget(self.session_time_label)
        
        # Total reps
        self.total_reps_label = QLabel("Total Reps: 0")
        group_layout.addWidget(self.total_reps_label)
        
        # Average accuracy
        self.avg_accuracy_label = QLabel("Avg Accuracy: 0%")
        group_layout.addWidget(self.avg_accuracy_label)
        
        # Best/Worst rep
        self.best_rep_label = QLabel("Best Rep: 0%")
        group_layout.addWidget(self.best_rep_label)
        
        self.worst_rep_label = QLabel("Worst Rep: 100%")
        group_layout.addWidget(self.worst_rep_label)
        
        # Session accuracy progress bar
        self.session_accuracy_bar = QProgressBar()
        self.session_accuracy_bar.setMaximum(100)
        self.session_accuracy_bar.setValue(0)
        self.session_accuracy_bar.setFormat("Session Score: %p%")
        group_layout.addWidget(self.session_accuracy_bar)
        
        layout.addWidget(group)
        
    def create_quality_breakdown_section(self, layout):
        group = QGroupBox("🏆 Rep Quality Breakdown")
        group_layout = QVBoxLayout(group)
        
        # Quality indicators
        self.excellent_count = QLabel("🥇 Excellent: 0")
        self.excellent_count.setStyleSheet("color: #4CAF50; font-weight: bold;")
        group_layout.addWidget(self.excellent_count)
        
        self.good_count = QLabel("🥈 Good: 0")
        self.good_count.setStyleSheet("color: #8BC34A; font-weight: bold;")
        group_layout.addWidget(self.good_count)
        
        self.fair_count = QLabel("🥉 Fair: 0")
        self.fair_count.setStyleSheet("color: #FF9800; font-weight: bold;")
        group_layout.addWidget(self.fair_count)
        
        self.poor_count = QLabel("❌ Poor: 0")
        self.poor_count.setStyleSheet("color: #F44336; font-weight: bold;")
        group_layout.addWidget(self.poor_count)
        
        # Quality trend
        self.quality_trend = QLabel("Trend: Getting started...")
        self.quality_trend.setStyleSheet("font-style: italic; color: #888;")
        group_layout.addWidget(self.quality_trend)
        
        layout.addWidget(group)
        
    def create_performance_section(self, layout):
        group = QGroupBox("📈 Performance Metrics")
        group_layout = QVBoxLayout(group)
        
        # Consistency score
        self.consistency_label = QLabel("Consistency: 0%")
        group_layout.addWidget(self.consistency_label)
        
        self.consistency_bar = QProgressBar()
        self.consistency_bar.setMaximum(100)
        group_layout.addWidget(self.consistency_bar)
        
        # Form stability
        self.stability_label = QLabel("Form Stability: Stable")
        group_layout.addWidget(self.stability_label)
        
        # Improvement suggestions
        self.suggestions_text = QTextEdit()
        self.suggestions_text.setMaximumHeight(100)
        self.suggestions_text.setPlaceholderText("Performance suggestions will appear here...")
        self.suggestions_text.setReadOnly(True)
        group_layout.addWidget(self.suggestions_text)
        
        layout.addWidget(group)
        
    def create_video_source_section(self, layout):
        group = QGroupBox("📹 Video Source")
        group_layout = QVBoxLayout(group)
        
        # Camera button
        self.camera_btn = QPushButton("📷 Use Camera")
        self.camera_btn.clicked.connect(self.setup_camera)
        self.camera_btn.setMinimumHeight(50)
        group_layout.addWidget(self.camera_btn)
        
        # Video file button
        self.video_btn = QPushButton("🎬 Load Video File")
        self.video_btn.clicked.connect(self.setup_video_file)
        self.video_btn.setMinimumHeight(50)
        group_layout.addWidget(self.video_btn)
        
        # Source info
        self.source_info = QLabel("No source selected")
        self.source_info.setAlignment(Qt.AlignCenter)
        self.source_info.setStyleSheet("color: #888; font-style: italic;")
        group_layout.addWidget(self.source_info)
        
        layout.addWidget(group)
        
    def create_exercise_section(self, layout):
        group = QGroupBox("🏋️ Exercise Selection")
        group_layout = QVBoxLayout(group)
        
        exercises = [
            ("Squats", ExerciseType.SQUAT, "🏃"),
            ("Push-ups", ExerciseType.PUSHUP, "💪"),
            ("Shoulder Press", ExerciseType.SHOULDER_PRESS, "🤸"),
            ("Dips", ExerciseType.DIPS, "🏋️"),
            ("Bench Press", ExerciseType.BENCH_PRESS, "💺"),
            ("Pull-ups", ExerciseType.PULL_UPS, "🏃‍♂️")
        ]
        
        self.exercise_buttons = []
        for name, exercise_type, emoji in exercises:
            btn = QPushButton(f"{emoji} {name}")
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda checked, ex=exercise_type, n=name: self.select_exercise(ex, n))
            btn.setCheckable(True)
            self.exercise_buttons.append(btn)
            group_layout.addWidget(btn)
            
        layout.addWidget(group)
        
    def create_controls_section(self, layout):
        group = QGroupBox("🎮 Controls")
        group_layout = QVBoxLayout(group)
        
        # Play/Pause
        self.play_pause_btn = QPushButton("▶️ Start")
        self.play_pause_btn.setMinimumHeight(50)
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setEnabled(False)
        group_layout.addWidget(self.play_pause_btn)
        
        # Reset button
        reset_btn = QPushButton("🔄 Reset Stats")
        reset_btn.setMinimumHeight(40)
        reset_btn.clicked.connect(self.reset_stats)
        group_layout.addWidget(reset_btn)
        
        # New session button
        new_session_btn = QPushButton("🆕 New Session")
        new_session_btn.setMinimumHeight(40)
        new_session_btn.clicked.connect(self.new_session)
        group_layout.addWidget(new_session_btn)
        
        # Settings button
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.setMinimumHeight(40)
        settings_btn.clicked.connect(self.show_settings)
        group_layout.addWidget(settings_btn)
        
        layout.addWidget(group)
        
    def setup_camera(self):
        try:
            # Try different camera indices
            for i in range(3):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        self.source_info.setText(f"📷 Camera {i} connected")
                        self.source_info.setStyleSheet("color: #4CAF50;")
                        self.play_pause_btn.setEnabled(True)
                        self.statusBar().showMessage("Camera ready - Select exercise and start")
                        return
                    cap.release()
            
            QMessageBox.warning(self, "Camera Error", "No working camera found!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Camera setup failed: {str(e)}")
            
    def setup_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.wmv *.mkv);;All Files (*)"
        )
        
        if file_path:
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        
                        # Get video info
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        
                        self.source_info.setText(f"🎬 Video loaded\n{frame_count} frames @ {fps:.1f} FPS")
                        self.source_info.setStyleSheet("color: #4CAF50;")
                        self.play_pause_btn.setEnabled(True)
                        self.video_controls.show()
                        self.progress_slider.setMaximum(frame_count)
                        self.statusBar().showMessage("Video loaded - Select exercise and start")
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                        return
                        
                cap.release()
                QMessageBox.warning(self, "Video Error", "Cannot read video file!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Video setup failed: {str(e)}")
                
    def select_exercise(self, exercise_type, exercise_name):
        # Uncheck all other buttons
        for btn in self.exercise_buttons:
            btn.setChecked(False)
            
        # Check selected button
        for btn in self.exercise_buttons:
            if exercise_name in btn.text():
                btn.setChecked(True)
                break
                
        self.form_corrector.select_exercise(exercise_type)
        self.exercise_info.setText(f"Selected: {exercise_name}")
        self.statusBar().showMessage(f"{exercise_name} selected - Ready to start")
        
    def toggle_playback(self):
        if self.video_thread is None or not self.video_thread.isRunning():
            self.start_video()
        else:
            if self.video_thread.paused:
                self.video_thread.resume()
                self.play_pause_btn.setText("⏸️ Pause")
            else:
                self.video_thread.pause()
                self.play_pause_btn.setText("▶️ Resume")
                
    def start_video(self):
        if self.cap and self.form_corrector.current_exercise:
            self.video_thread = VideoThread(self.cap, self.form_corrector)
            self.video_thread.frame_ready.connect(self.update_video_display)
            self.video_thread.stats_updated.connect(self.update_stats)
            self.video_thread.start()
            self.play_pause_btn.setText("⏸️ Pause")
            self.statusBar().showMessage("Analysis running...")
            
            # Start session timer
            self.session_timer = QTimer()
            self.session_timer.timeout.connect(self.update_session_time)
            self.session_timer.start(1000)  # Update every second
        else:
            QMessageBox.information(self, "Setup Required", 
                                  "Please select both a video source and exercise type.")
            
    def update_video_display(self, frame):
        # Convert frame to Qt format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit display
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_stats(self, stats):
        # Update current rep info
        self.rep_display.setText(f"Reps: {stats.get('reps', 0)}")
        
        current_accuracy = stats.get('current_rep_accuracy', 0)
        self.current_accuracy_label.setText(f"Current Rep: {current_accuracy}%")
        
        # Update rep quality
        quality = stats.get('rep_quality', 'Ready')
        self.quality_label.setText(f"Quality: {quality}")
        
        # Color code quality
        quality_colors = {
            'Excellent': '#4CAF50',
            'Good': '#8BC34A', 
            'Fair': '#FF9800',
            'Poor': '#F44336',
            'Ready': '#888'
        }
        quality_color = quality_colors.get(quality, '#888')
        self.quality_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {quality_color};")
        
        # Update phase
        phase = stats.get('phase', 'Ready')
        self.phase_label.setText(f"Phase: {phase}")
        
        # Update range of motion
        rom = stats.get('range_of_motion', 0)
        self.rom_progress.setValue(int(rom))
        
        # Update live feedback
        feedback = stats.get('feedback', [])
        if feedback:
            self.live_feedback.setText('\n'.join([f"• {msg}" for msg in feedback]))
        else:
            self.live_feedback.setText("Form looks good! 👍")
            
        # Update session statistics when a rep is completed
        if stats.get('rep_completed', False):
            rep_accuracy = stats.get('completed_rep_accuracy', 0)
            rep_quality_enum = self.get_quality_from_accuracy(rep_accuracy)
            self.workout_session.add_rep(rep_accuracy, rep_quality_enum)
            self.update_session_display()
            
        # Update stability and movement info
        stability = stats.get('stability', 'Stable')
        self.stability_label.setText(f"Form Stability: {stability}")
        
    def get_quality_from_accuracy(self, accuracy):
        if accuracy >= 90:
            return RepQuality.EXCELLENT
        elif accuracy >= 75:
            return RepQuality.GOOD
        elif accuracy >= 60:
            return RepQuality.FAIR
        else:
            return RepQuality.POOR
            
    def update_session_display(self):
        # Update session stats
        session = self.workout_session
        self.total_reps_label.setText(f"Total Reps: {session.total_reps}")
        self.avg_accuracy_label.setText(f"Avg Accuracy: {session.total_accuracy:.1f}%")
        self.best_rep_label.setText(f"Best Rep: {session.best_rep_accuracy:.1f}%")
        
        if session.total_reps > 0:
            self.worst_rep_label.setText(f"Worst Rep: {session.worst_rep_accuracy:.1f}%")
        
        self.session_accuracy_bar.setValue(int(session.total_accuracy))
        
        # Update quality breakdown
        quality_dist = session.get_quality_distribution()
        self.excellent_count.setText(f"🥇 Excellent: {quality_dist['Excellent']}")
        self.good_count.setText(f"🥈 Good: {quality_dist['Good']}")
        self.fair_count.setText(f"🥉 Fair: {quality_dist['Fair']}")
        self.poor_count.setText(f"❌ Poor: {quality_dist['Poor']}")
        
        # Calculate consistency
        if session.total_reps >= 3:
            recent_qualities = session.rep_qualities[-5:]  # Last 5 reps
            consistency = self.calculate_consistency(recent_qualities)
            self.consistency_label.setText(f"Consistency: {consistency:.1f}%")
            self.consistency_bar.setValue(int(consistency))
            
        # Update suggestions
        self.update_suggestions()
        
    def calculate_consistency(self, qualities):
        if len(qualities) < 2:
            return 0
            
        quality_scores = {
            RepQuality.EXCELLENT: 4,
            RepQuality.GOOD: 3,
            RepQuality.FAIR: 2,
            RepQuality.POOR: 1
        }
        
        scores = [quality_scores.get(q, 1) for q in qualities]
        if len(scores) < 2:
            return 0
            
        variance = np.var(scores)
        max_variance = 2.25  # Maximum possible variance for scores 1-4
        consistency = max(0, (1 - variance / max_variance)) * 100
        return consistency
        
    def update_suggestions(self):
        session = self.workout_session
        suggestions = []
        
        if session.total_reps >= 5:
            quality_dist = session.get_quality_distribution()
            poor_percentage = (quality_dist['Poor'] / session.total_reps) * 100
            
            if poor_percentage > 30:
                suggestions.append("Focus on form over speed")
                
            if session.total_accuracy < 70:
                suggestions.append("Review exercise technique")
                
            recent_qualities = session.rep_qualities[-3:]
            if len(recent_qualities) == 3 and all(q == RepQuality.POOR for q in recent_qualities):
                suggestions.append("Take a break - form is declining")
                
            if session.best_rep_accuracy - session.worst_rep_accuracy > 40:
                suggestions.append("Work on consistency")
                
        if not suggestions:
            suggestions.append("Keep up the great work!")
            
        self.suggestions_text.setText('\n'.join([f"• {s}" for s in suggestions]))
        
    def update_session_time(self):
        duration = datetime.now() - self.workout_session.start_time
        minutes, seconds = divmod(duration.total_seconds(), 60)
        self.session_time_label.setText(f"Duration: {int(minutes):02d}:{int(seconds):02d}")
        
    def reset_stats(self):
        self.form_corrector.reset_stats()
        self.rep_display.setText("Reps: 0")
        self.current_accuracy_label.setText("Current Rep: 0%")
        self.quality_label.setText("Quality: Ready")
        self.phase_label.setText("Phase: Ready")
        self.rom_progress.setValue(0)
        self.live_feedback.clear()
        self.statusBar().showMessage("Stats reset")
        
    def new_session(self):
        self.reset_stats()
        self.workout_session = WorkoutSession()
        self.update_session_display()
        
        # Reset all session displays
        self.session_time_label.setText("Duration: 00:00")
        self.total_reps_label.setText("Total Reps: 0")
        self.avg_accuracy_label.setText("Avg Accuracy: 0%")
        self.best_rep_label.setText("Best Rep: 0%")
        self.worst_rep_label.setText("Worst Rep: 100%")
        self.session_accuracy_bar.setValue(0)
        
        self.excellent_count.setText("🥇 Excellent: 0")
        self.good_count.setText("🥈 Good: 0")
        self.fair_count.setText("🥉 Fair: 0")
        self.poor_count.setText("❌ Poor: 0")
        
        self.consistency_label.setText("Consistency: 0%")
        self.consistency_bar.setValue(0)
        self.suggestions_text.clear()
        
        self.statusBar().showMessage("New session started")
        
    def save_session(self):
        if self.workout_session.total_reps == 0:
            QMessageBox.information(self, "No Data", "No workout data to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", 
            f"workout_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                session_data = {
                    'start_time': self.workout_session.start_time.isoformat(),
                    'total_reps': self.workout_session.total_reps,
                    'total_accuracy': self.workout_session.total_accuracy,
                    'best_rep_accuracy': self.workout_session.best_rep_accuracy,
                    'worst_rep_accuracy': self.workout_session.worst_rep_accuracy,
                    'rep_qualities': [q.value if isinstance(q, RepQuality) else q for q in self.workout_session.rep_qualities],
                    'exercise_type': self.form_corrector.current_exercise.name if self.form_corrector.current_exercise else None,
                    'settings': self.settings
                }
                
                with open(file_path, 'w') as f:
                    json.dump(session_data, f, indent=2)
                    
                QMessageBox.information(self, "Success", f"Session saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save session: {str(e)}")
                
    def load_session(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    session_data = json.load(f)
                    
                # Create new session with loaded data
                self.workout_session = WorkoutSession()
                self.workout_session.start_time = datetime.fromisoformat(session_data['start_time'])
                self.workout_session.total_reps = session_data['total_reps']
                self.workout_session.total_accuracy = session_data['total_accuracy']
                self.workout_session.best_rep_accuracy = session_data['best_rep_accuracy']
                self.workout_session.worst_rep_accuracy = session_data['worst_rep_accuracy']
                self.workout_session.rep_qualities = [RepQuality(q) for q in session_data['rep_qualities']]
                
                # Update displays
                self.update_session_display()
                
                QMessageBox.information(self, "Success", f"Session loaded from {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load session: {str(e)}")
                
    def export_data(self):
        if self.workout_session.total_reps == 0:
            QMessageBox.information(self, "No Data", "No workout data to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", 
            f"workout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("AI FORM CORRECTOR - WORKOUT REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Session Date: {self.workout_session.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Exercise: {self.form_corrector.current_exercise.name if self.form_corrector.current_exercise else 'Unknown'}\n")
                    f.write(f"Duration: {datetime.now() - self.workout_session.start_time}\n\n")
                    
                    f.write("PERFORMANCE SUMMARY\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Total Reps: {self.workout_session.total_reps}\n")
                    f.write(f"Average Accuracy: {self.workout_session.total_accuracy:.1f}%\n")
                    f.write(f"Best Rep: {self.workout_session.best_rep_accuracy:.1f}%\n")
                    f.write(f"Worst Rep: {self.workout_session.worst_rep_accuracy:.1f}%\n\n")
                    
                    quality_dist = self.workout_session.get_quality_distribution()
                    f.write("QUALITY BREAKDOWN\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Excellent: {quality_dist['Excellent']}\n")
                    f.write(f"Good: {quality_dist['Good']}\n")
                    f.write(f"Fair: {quality_dist['Fair']}\n")
                    f.write(f"Poor: {quality_dist['Poor']}\n\n")
                    
                    if len(self.workout_session.rep_qualities) >= 3:
                        consistency = self.calculate_consistency(self.workout_session.rep_qualities[-5:])
                        f.write(f"Consistency Score: {consistency:.1f}%\n\n")
                    
                QMessageBox.information(self, "Success", f"Report exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")
                
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
            
    def load_settings(self):
        try:
            if os.path.exists('form_corrector_settings.json'):
                with open('form_corrector_settings.json', 'r') as f:
                    self.settings.update(json.load(f))
        except:
            pass
            
    def save_settings(self):
        try:
            with open('form_corrector_settings.json', 'w') as f:
                json.dump(self.settings, f, indent=2)
        except:
            pass
        
    def show_settings(self):
        dialog = AdvancedSettingsDialog(self, self.settings)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.get_settings()
            self.save_settings()
            
            # Apply settings to form corrector
            self.form_corrector.update_settings(self.settings)
        
    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        if self.cap:
            self.cap.release()
            
        # Auto-save session if enabled
        if self.settings.get('auto_save_sessions', True) and self.workout_session.total_reps > 0:
            try:
                session_data = {
                    'start_time': self.workout_session.start_time.isoformat(),
                    'total_reps': self.workout_session.total_reps,
                    'total_accuracy': self.workout_session.total_accuracy,
                    'best_rep_accuracy': self.workout_session.best_rep_accuracy,
                    'worst_rep_accuracy': self.workout_session.worst_rep_accuracy,
                    'rep_qualities': [q.value if isinstance(q, RepQuality) else q for q in self.workout_session.rep_qualities],
                    'exercise_type': self.form_corrector.current_exercise.name if self.form_corrector.current_exercise else None
                }
                
                os.makedirs('auto_saved_sessions', exist_ok=True)
                filename = f"auto_saved_sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2)
            except:
                pass
                
        event.accept()
        
    def get_dark_theme(self):
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555;
            border-radius: 10px;
            margin: 10px 0px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QPushButton {
            background-color: #404040;
            border: 2px solid #555;
            border-radius: 8px;
            padding: 8px;
            font-size: 14px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #505050;
            border-color: #777;
        }
        
        QPushButton:pressed {
            background-color: #353535;
        }
        
        QPushButton:checked {
            background-color: #4CAF50;
            border-color: #45a049;
        }
        
        QFrame {
            background-color: #353535;
            border-radius: 10px;
            margin: 5px;
        }
        
        QLabel {
            color: #ffffff;
        }
        
        QTextEdit {
            background-color: #404040;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 5px;
        }
        
        QProgressBar {
            border: 2px solid grey;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 3px;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
            margin: 2px 0;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 3px;
        }
        
        QStatusBar {
            background-color: #404040;
            color: #ffffff;
        }
        
        QMenuBar {
            background-color: #404040;
            color: #ffffff;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 4px 8px;
        }
        
        QMenuBar::item:selected {
            background-color: #505050;
        }
        
        QMenu {
            background-color: #404040;
            color: #ffffff;
            border: 1px solid #555;
        }
        
        QMenu::item:selected {
            background-color: #505050;
        }
        """

class AdvancedSettingsDialog(QDialog):
    def __init__(self, parent, current_settings):
        super().__init__(parent)
        self.settings = current_settings.copy()
        self.setWindowTitle("⚙️ Advanced Settings")
        self.setModal(True)
        self.setFixedSize(500, 600)
        
        layout = QVBoxLayout(self)
        
        # Detection settings
        detection_group = QGroupBox("🎯 Detection Settings")
        detection_layout = QVBoxLayout(detection_group)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Detection Confidence:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(50, 95)
        self.confidence_slider.setValue(int(self.settings['confidence_threshold'] * 100))
        self.confidence_label = QLabel(f"{self.settings['confidence_threshold']:.2f}")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_label.setText(f"{v/100:.2f}")
        )
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(self.confidence_label)
        detection_layout.addLayout(conf_layout)
        
        # Smoothing factor
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smoothing Factor:"))
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(50, 95)
        self.smoothing_slider.setValue(int(self.settings['smoothing_factor'] * 100))
        self.smoothing_label = QLabel(f"{self.settings['smoothing_factor']:.2f}")
        self.smoothing_slider.valueChanged.connect(
            lambda v: self.smoothing_label.setText(f"{v/100:.2f}")
        )
        smooth_layout.addWidget(self.smoothing_slider)
        smooth_layout.addWidget(self.smoothing_label)
        detection_layout.addLayout(smooth_layout)
        
        # Rep hold frames
        hold_layout = QHBoxLayout()
        hold_layout.addWidget(QLabel("Rep Hold Frames:"))
        self.hold_frames_spin = QSpinBox()
        self.hold_frames_spin.setRange(2, 10)
        self.hold_frames_spin.setValue(self.settings['rep_hold_frames'])
        hold_layout.addWidget(self.hold_frames_spin)
        detection_layout.addLayout(hold_layout)
        
        layout.addWidget(detection_group)
        
        # Session settings
        session_group = QGroupBox("💾 Session Settings")
        session_layout = QVBoxLayout(session_group)
        
        self.auto_save_check = QCheckBox("Auto-save sessions on exit")
        self.auto_save_check.setChecked(self.settings['auto_save_sessions'])
        session_layout.addWidget(self.auto_save_check)
        
        layout.addWidget(session_group)
        
        # Exercise-specific settings
        exercise_group = QGroupBox("🏋️ Exercise Thresholds")
        exercise_layout = QVBoxLayout(exercise_group)
        
        # Add exercise-specific threshold controls here
        info_label = QLabel("Exercise-specific thresholds can be adjusted here in future versions.")
        info_label.setStyleSheet("color: #888; font-style: italic;")
        exercise_layout.addWidget(info_label)
        
        layout.addWidget(exercise_group)
        
        # Tips section
        tips_group = QGroupBox("💡 Tips for Better Results")
        tips_layout = QVBoxLayout(tips_group)
        
        tips = [
            "📹 Position camera 3-6 feet away for best tracking",
            "💡 Ensure good lighting without harsh shadows", 
            "👕 Wear fitted, contrasting clothing",
            "📐 Side view works best for squats and dips",
            "🎯 Front view recommended for shoulder press",
            "⚡ Move slowly and deliberately for better accuracy",
            "🏃 Stay within camera frame throughout exercise",
            "📏 Maintain consistent distance from camera"
        ]
        
        for tip in tips:
            label = QLabel(tip)
            label.setWordWrap(True)
            tips_layout.addWidget(label)
            
        layout.addWidget(tips_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("✅ OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
    def reset_to_defaults(self):
        self.confidence_slider.setValue(70)
        self.smoothing_slider.setValue(80)
        self.hold_frames_spin.setValue(5)
        self.auto_save_check.setChecked(True)
        
    def get_settings(self):
        return {
            'confidence_threshold': self.confidence_slider.value() / 100,
            'smoothing_factor': self.smoothing_slider.value() / 100,
            'rep_hold_frames': self.hold_frames_spin.value(),
            'auto_save_sessions': self.auto_save_check.isChecked()
        }

class AIFormCorrector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise state
        self.current_exercise = None
        self.rep_count = 0
        self.current_rep_accuracy = 0
        self.completed_rep_accuracy = 0
        self.is_down_position = False
        self.rep_completed = False
        
        # Enhanced tracking
        self.angle_history = deque(maxlen=15)
        self.accuracy_history = deque(maxlen=10)
        self.down_hold_counter = 0
        self.up_hold_counter = 0
        self.last_rep_time = 0
        self.movement_phase = "Ready"
        self.range_of_motion = 0
        self.stability_score = 100
        
        # Settings
        self.settings = {
            'confidence_threshold': 0.7,
            'smoothing_factor': 0.8,
            'rep_hold_frames': 5
        }
        
        # Body joints mapping
        self.joints = {
            11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER',
            13: 'LEFT_ELBOW', 14: 'RIGHT_ELBOW',
            15: 'LEFT_WRIST', 16: 'RIGHT_WRIST',
            23: 'LEFT_HIP', 24: 'RIGHT_HIP',
            25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
            27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE'
        }
        
        # Exercise-specific parameters
        self.exercise_params = {
            ExerciseType.SQUAT: {
                'down_threshold': 110,
                'up_threshold': 150,
                'optimal_range': (90, 120),
                'max_asymmetry': 20
            },
            ExerciseType.PUSHUP: {
                'down_threshold': 120,
                'up_threshold': 160,
                'optimal_range': (90, 140),
                'max_asymmetry': 25
            },
            ExerciseType.SHOULDER_PRESS: {
                'down_threshold': 100,
                'up_threshold': 160,
                'optimal_range': (90, 110),
                'max_asymmetry': 15
            },
            ExerciseType.DIPS: {
                'down_threshold': 90,
                'up_threshold': 160,
                'optimal_range': (70, 100),
                'max_asymmetry': 20
            },
            ExerciseType.BENCH_PRESS: {
                'down_threshold': 90,
                'up_threshold': 160,
                'optimal_range': (80, 100),
                'max_asymmetry': 15
            },
            ExerciseType.PULL_UPS: {
                'down_threshold': -20,
                'up_threshold': 20,
                'optimal_range': (-40, -10),
                'max_asymmetry': 30
            }
        }
        
    def update_settings(self, settings):
        self.settings.update(settings)
        # Update MediaPipe pose with new confidence threshold
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.settings['confidence_threshold'],
            min_tracking_confidence=self.settings['confidence_threshold']
        )
        
    def select_exercise(self, exercise_type):
        self.current_exercise = exercise_type
        self.reset_stats()
        
    def reset_stats(self):
        self.rep_count = 0
        self.current_rep_accuracy = 0
        self.completed_rep_accuracy = 0
        self.is_down_position = False
        self.rep_completed = False
        self.angle_history.clear()
        self.accuracy_history.clear()
        self.down_hold_counter = 0
        self.up_hold_counter = 0
        self.movement_phase = "Ready"
        self.range_of_motion = 0
        self.stability_score = 100
        
    def process_frame(self, frame):
        """Process frame and return annotated frame with enhanced stats"""
        if frame is None:
            return frame, {}
            
        # Flip for mirror effect (for camera)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb_frame)
        joints = self.extract_joints(results, frame.shape)
        
        # Reset rep completed flag
        self.rep_completed = False
        
        # Analyze exercise
        feedback, current_accuracy = self.analyze_exercise(joints)
        
        # Update current rep accuracy with smoothing
        if len(self.accuracy_history) > 0:
            smoothing = self.settings['smoothing_factor']
            self.current_rep_accuracy = (smoothing * self.current_rep_accuracy + 
                                       (1 - smoothing) * current_accuracy)
        else:
            self.current_rep_accuracy = current_accuracy
            
        self.accuracy_history.append(current_accuracy)
        
        # Calculate stability
        self.calculate_stability()
        
        # Draw pose landmarks with enhanced visualization
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add custom overlays
            self.add_custom_overlays(frame, joints)
            
        # Determine rep quality
        rep_quality = self.get_rep_quality(self.current_rep_accuracy)
        
        # Prepare enhanced stats
        stats = {
            'reps': self.rep_count,
            'current_rep_accuracy': int(self.current_rep_accuracy),
            'completed_rep_accuracy': int(self.completed_rep_accuracy),
            'rep_quality': rep_quality,
            'phase': self.movement_phase,
            'range_of_motion': self.range_of_motion,
            'stability': "Stable" if self.stability_score > 70 else "Shaky",
            'feedback': feedback,
            'rep_completed': self.rep_completed
        }
        
        return frame, stats
        
    def add_custom_overlays(self, frame, joints):
        """Add custom visual overlays to the frame"""
        if not joints or self.current_exercise is None:
            return
            
        # Add angle indicators for key joints
        if self.current_exercise == ExerciseType.SQUAT:
            self.draw_angle_indicator(frame, joints, 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE')
            self.draw_angle_indicator(frame, joints, 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')
        elif self.current_exercise == ExerciseType.PUSHUP:
            self.draw_angle_indicator(frame, joints, 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST')
            self.draw_angle_indicator(frame, joints, 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST')
            
    def draw_angle_indicator(self, frame, joints, joint1, joint2, joint3):
        """Draw angle indicator on frame"""
        try:
            p1 = self.get_joint_coords(joints, joint1)
            p2 = self.get_joint_coords(joints, joint2)
            p3 = self.get_joint_coords(joints, joint3)
            
            if all(coord != [0, 0] for coord in [p1, p2, p3]):
                angle = self.calculate_angle(p1, p2, p3)
                
                # Draw angle arc (simplified as a circle for now)
                cv2.circle(frame, (int(p2[0]), int(p2[1])), 30, (0, 255, 0), 2)
                cv2.putText(frame, f"{int(angle)}°", (int(p2[0]) + 35, int(p2[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except:
            pass
            
    def calculate_stability(self):
        """Calculate form stability based on recent angle variations"""
        if len(self.angle_history) < 5:
            self.stability_score = 100
            return
            
        recent_angles = list(self.angle_history)[-5:]
        variance = np.var(recent_angles)
        
        # Convert variance to stability score (0-100)
        # Lower variance = higher stability
        max_variance = 400  # Adjust based on typical variance observed
        self.stability_score = max(0, 100 - (variance / max_variance * 100))
        
    def get_rep_quality(self, accuracy):
        """Determine rep quality based on accuracy"""
        if accuracy >= 90:
            return "Excellent"
        elif accuracy >= 75:
            return "Good"
        elif accuracy >= 60:
            return "Fair"
        else:
            return "Poor"
            
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            a, b, c = np.array(a), np.array(b), np.array(c)
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180.0 else angle
        except:
            return 0
            
    def get_joint_coords(self, joints, joint_name):
        """Get joint coordinates as [x, y]"""
        if joints and joint_name in joints:
            return [joints[joint_name]['x'], joints[joint_name]['y']]
        return [0, 0]
        
    def extract_joints(self, results, image_shape):
        """Extract joint coordinates"""
        if not results.pose_landmarks:
            return None
        
        joints = {}
        h, w = image_shape[:2]
        
        for idx, name in self.joints.items():
            try:
                landmark = results.pose_landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    joints[name] = {
                        'x': int(landmark.x * w),
                        'y': int(landmark.y * h),
                        'visibility': landmark.visibility
                    }
            except Exception as e:
                continue
        
        return joints if joints else None
        
    def count_rep_with_stability(self, current_angle, exercise_type):
        """Enhanced rep counting with stability checks and phase detection"""
        if current_angle == 0 or exercise_type not in self.exercise_params:
            return
            
        params = self.exercise_params[exercise_type]
        current_time = time.time()
        
        # Prevent too rapid counting
        if current_time - self.last_rep_time < 0.8:
            return
            
        self.angle_history.append(current_angle)
        
        # Calculate range of motion
        if len(self.angle_history) >= 5:
            recent_angles = list(self.angle_history)[-5:]
            angle_range = max(recent_angles) - min(recent_angles)
            self.range_of_motion = min(100, (angle_range / 80) * 100)  # Normalize to 0-100
        
        # Phase detection and rep counting
        required_hold = self.settings['rep_hold_frames']
        
        if current_angle <= params['down_threshold']:
            self.down_hold_counter += 1
            self.up_hold_counter = 0
            self.movement_phase = "Descending"
            
            if self.down_hold_counter >= required_hold and not self.is_down_position:
                self.is_down_position = True
                self.movement_phase = "Bottom Hold"
        
        elif current_angle >= params['up_threshold']:
            self.up_hold_counter += 1
            self.down_hold_counter = 0
            self.movement_phase = "Ascending"
            
            if self.up_hold_counter >= required_hold and self.is_down_position:
                # Rep completed!
                self.is_down_position = False
                self.rep_count += 1
                self.rep_completed = True
                self.completed_rep_accuracy = self.current_rep_accuracy
                self.last_rep_time = current_time
                self.movement_phase = "Top Hold"
        
        else:
            # In transition phase
            self.down_hold_counter = max(0, self.down_hold_counter - 1)
            self.up_hold_counter = max(0, self.up_hold_counter - 1)
            if self.is_down_position:
                self.movement_phase = "Ascending"
            else:
                self.movement_phase = "Descending"
            
    def analyze_exercise(self, joints):
        """Enhanced exercise analysis with detailed feedback"""
        if not joints or self.current_exercise is None:
            return [], 0
        
        try:
            if self.current_exercise == ExerciseType.SQUAT:
                return self.analyze_squat(joints)
            elif self.current_exercise == ExerciseType.PUSHUP:
                return self.analyze_pushup(joints)
            elif self.current_exercise == ExerciseType.SHOULDER_PRESS:
                return self.analyze_shoulder_press(joints)
            elif self.current_exercise == ExerciseType.DIPS:
                return self.analyze_dips(joints)
            elif self.current_exercise == ExerciseType.BENCH_PRESS:
                return self.analyze_bench_press(joints)
            elif self.current_exercise == ExerciseType.PULL_UPS:
                return self.analyze_pull_ups(joints)
        except Exception as e:
            pass
        
        return [], 0
        
    def analyze_squat(self, joints):
        """Enhanced squat analysis with detailed form feedback"""
        feedback, accuracy = [], 100
        
        try:
            # Get joint coordinates
            lh = self.get_joint_coords(joints, 'LEFT_HIP')
            lk = self.get_joint_coords(joints, 'LEFT_KNEE')
            la = self.get_joint_coords(joints, 'LEFT_ANKLE')
            rh = self.get_joint_coords(joints, 'RIGHT_HIP')
            rk = self.get_joint_coords(joints, 'RIGHT_KNEE')
            ra = self.get_joint_coords(joints, 'RIGHT_ANKLE')
            ls = self.get_joint_coords(joints, 'LEFT_SHOULDER')
            rs = self.get_joint_coords(joints, 'RIGHT_SHOULDER')
            
            if all(coord != [0, 0] for coord in [lh, lk, la, rh, rk, ra]):
                # Calculate knee angles
                left_angle = self.calculate_angle(lh, lk, la)
                right_angle = self.calculate_angle(rh, rk, ra)
                avg_angle = (left_angle + right_angle) / 2
                
                # Count reps with enhanced stability
                self.count_rep_with_stability(avg_angle, ExerciseType.SQUAT)
                
                params = self.exercise_params[ExerciseType.SQUAT]
                
                # Form analysis
                asymmetry = abs(left_angle - right_angle)
                if asymmetry > params['max_asymmetry']:
                    feedback.append(f"Keep knees aligned (asymmetry: {asymmetry:.1f}°)")
                    accuracy -= min(20, asymmetry)
                
                # Depth analysis
                if self.is_down_position:
                    if avg_angle > params['optimal_range'][1]:
                        feedback.append("Go deeper - squat below parallel")
                        accuracy -= 15
                    elif avg_angle < params['optimal_range'][0]:
                        feedback.append("Don't go too deep - risk of injury")
                        accuracy -= 10
                
                # Knee tracking (knees should track over toes)
                if lk[0] < la[0] - 20 or rk[0] < ra[0] - 20:
                    feedback.append("Keep knees over toes")
                    accuracy -= 12
                
                # Torso posture (shoulders should stay relatively upright)
                if ls != [0, 0] and rs != [0, 0]:
                    shoulder_y = (ls[1] + rs[1]) / 2
                    hip_y = (lh[1] + rh[1]) / 2
                    if shoulder_y > hip_y + 100:  # Too much forward lean
                        feedback.append("Keep chest up - avoid forward lean")
                        accuracy -= 10
                
                # Range of motion bonus/penalty
                if self.range_of_motion < 40:
                    feedback.append("Increase range of motion")
                    accuracy -= 15
                elif self.range_of_motion > 80:
                    accuracy += 5  # Bonus for good ROM
                    
            else:
                feedback.append("Position yourself better in frame")
                accuracy = 0
                
        except Exception as e:
            feedback.append("Analysis error - check positioning")
            accuracy = 0
            
        return feedback, max(0, min(100, accuracy))
        
    def analyze_pushup(self, joints):
        """Enhanced pushup analysis"""
        feedback, accuracy = [], 100
        
        try:
            # Get joint coordinates
            ls = self.get_joint_coords(joints, 'LEFT_SHOULDER')
            le = self.get_joint_coords(joints, 'LEFT_ELBOW')
            lw = self.get_joint_coords(joints, 'LEFT_WRIST')
            rs = self.get_joint_coords(joints, 'RIGHT_SHOULDER')
            re = self.get_joint_coords(joints, 'RIGHT_ELBOW')
            rw = self.get_joint_coords(joints, 'RIGHT_WRIST')
            lh = self.get_joint_coords(joints, 'LEFT_HIP')
            rh = self.get_joint_coords(joints, 'RIGHT_HIP')
            
            if all(coord != [0, 0] for coord in [ls, le, lw, rs, re, rw]):
                # Calculate elbow angles
                left_angle = self.calculate_angle(ls, le, lw)
                right_angle = self.calculate_angle(rs, re, rw)
                avg_angle = (left_angle + right_angle) / 2
                
                self.count_rep_with_stability(avg_angle, ExerciseType.PUSHUP)
                
                params = self.exercise_params[ExerciseType.PUSHUP]
                
                # Arm symmetry
                asymmetry = abs(left_angle - right_angle)
                if asymmetry > params['max_asymmetry']:
                    feedback.append("Balance both arms equally")
                    accuracy -= min(15, asymmetry * 0.6)
                
                # Depth analysis
                if self.is_down_position:
                    if avg_angle > params['optimal_range'][1]:
                        feedback.append("Lower your chest more")
                        accuracy -= 18
                    elif avg_angle < params['optimal_range'][0]:
                        feedback.append("Don't go too low")
                        accuracy -= 8
                
                # Body alignment (plank position)
                if lh != [0, 0] and rh != [0, 0] and ls != [0, 0] and rs != [0, 0]:
                    shoulder_y = (ls[1] + rs[1]) / 2
                    hip_y = (lh[1] + rh[1]) / 2
                    alignment_diff = abs(shoulder_y - hip_y)
                    
                    if alignment_diff > 50:
                        feedback.append("Keep body in straight line")
                        accuracy -= min(12, alignment_diff * 0.2)
                
                # Elbow flare check
                elbow_shoulder_distance = abs(le[0] - ls[0]) + abs(re[0] - rs[0])
                if elbow_shoulder_distance > 200:  # Elbows too wide
                    feedback.append("Keep elbows closer to body")
                    accuracy -= 10
                    
                # Range of motion
                if self.range_of_motion < 35:
                    feedback.append("Increase range of motion")
                    accuracy -= 12
                elif self.range_of_motion > 70:
                    accuracy += 3
                    
            else:
                feedback.append("Position yourself better in frame")
                accuracy = 0
                
        except Exception:
            feedback.append("Analysis error - check positioning")
            accuracy = 0
            
        return feedback, max(0, min(100, accuracy))
        
    def analyze_shoulder_press(self, joints):
        """Enhanced shoulder press analysis"""
        feedback, accuracy = [], 100
        
        try:
            ls = self.get_joint_coords(joints, 'LEFT_SHOULDER')
            le = self.get_joint_coords(joints, 'LEFT_ELBOW')
            lw = self.get_joint_coords(joints, 'LEFT_WRIST')
            rs = self.get_joint_coords(joints, 'RIGHT_SHOULDER')
            re = self.get_joint_coords(joints, 'RIGHT_ELBOW')
            rw = self.get_joint_coords(joints, 'RIGHT_WRIST')
            
            if all(coord != [0, 0] for coord in [ls, le, lw, rs, re, rw]):
                left_angle = self.calculate_angle(ls, le, lw)
                right_angle = self.calculate_angle(rs, re, rw)
                avg_angle = (left_angle + right_angle) / 2
                
                self.count_rep_with_stability(avg_angle, ExerciseType.SHOULDER_PRESS)
                
                params = self.exercise_params[ExerciseType.SHOULDER_PRESS]
                
                # Symmetry check
                asymmetry = abs(left_angle - right_angle)
                if asymmetry > params['max_asymmetry']:
                    feedback.append("Press both arms equally")
                    accuracy -= min(18, asymmetry * 1.2)
                
                # Full extension check
                if not self.is_down_position:  # At the top
                    if lw[1] > ls[1] or rw[1] > rs[1]:
                        feedback.append("Press weights fully overhead")
                        accuracy -= 15
                
                # Starting position check
                if self.is_down_position:
                    if avg_angle < params['optimal_range'][0]:
                        feedback.append("Don't lower weights too much")
                        accuracy -= 10
                
                # Wrist alignment
                wrist_alignment = abs(lw[0] - ls[0]) + abs(rw[0] - rs[0])
                if wrist_alignment > 100:
                    feedback.append("Keep wrists aligned over shoulders")
                    accuracy -= 8
                    
                if self.range_of_motion < 30:
                    feedback.append("Increase range of motion")
                    accuracy -= 15
                    
            else:
                feedback.append("Face camera for better tracking")
                accuracy = 0
                
        except Exception:
            accuracy = 0
            
        return feedback, max(0, min(100, accuracy))
        
    def analyze_dips(self, joints):
        """Enhanced dips analysis"""
        feedback, accuracy = [], 100
        
        try:
            ls = self.get_joint_coords(joints, 'LEFT_SHOULDER')
            le = self.get_joint_coords(joints, 'LEFT_ELBOW')
            lw = self.get_joint_coords(joints, 'LEFT_WRIST')
            rs = self.get_joint_coords(joints, 'RIGHT_SHOULDER')
            
            if all(coord != [0, 0] for coord in [ls, le, lw]):
                left_angle = self.calculate_angle(ls, le, lw)
                self.count_rep_with_stability(left_angle, ExerciseType.DIPS)
                
                params = self.exercise_params[ExerciseType.DIPS]
                
                # Depth check
                if self.is_down_position:
                    if left_angle < params['optimal_range'][0]:
                        feedback.append("Don't dip too low - risk of injury")
                        accuracy -= 12
                    elif left_angle > params['optimal_range'][1]:
                        feedback.append("Lower yourself more")
                        accuracy -= 20
                
                # Body position (should be relatively vertical)
                if le[1] < ls[1] - 80:  # Elbow too high relative to shoulder
                    feedback.append("Keep elbows close to body")
                    accuracy -= 10
                
                # Range of motion
                if self.range_of_motion < 25:
                    feedback.append("Increase depth gradually")
                    accuracy -= 15
                elif self.range_of_motion > 60:
                    accuracy += 5
                    
            else:
                feedback.append("Position yourself better for side view")
                accuracy = 0
                
        except Exception:
            accuracy = 0
            
        return feedback, max(0, min(100, accuracy))
        
    def analyze_bench_press(self, joints):
        """Enhanced bench press analysis"""
        feedback, accuracy = [], 100
        
        try:
            ls = self.get_joint_coords(joints, 'LEFT_SHOULDER')
            le = self.get_joint_coords(joints, 'LEFT_ELBOW')
            lw = self.get_joint_coords(joints, 'LEFT_WRIST')
            
            if all(coord != [0, 0] for coord in [ls, le, lw]):
                left_angle = self.calculate_angle(ls, le, lw)
                self.count_rep_with_stability(left_angle, ExerciseType.BENCH_PRESS)
                
                params = self.exercise_params[ExerciseType.BENCH_PRESS]
                
                # Bar path (elbow should not go too far below shoulder level)
                if self.is_down_position and le[1] > ls[1] + 60:
                    feedback.append("Don't lower bar too far")
                    accuracy -= 18
                
                # Full extension check
                if not self.is_down_position and left_angle < 160:
                    feedback.append("Press bar to full extension")
                    accuracy -= 12
                
                # Elbow position
                if abs(le[0] - ls[0]) > 80:  # Elbow too far from shoulder line
                    feedback.append("Keep elbows in proper position")
                    accuracy -= 10
                
                if self.range_of_motion < 20:
                    feedback.append("Increase range of motion")
                    accuracy -= 15
                    
            else:
                feedback.append("Position for side view tracking")
                accuracy = 0
                
        except Exception:
            accuracy = 0
            
        return feedback, max(0, min(100, accuracy))
        
    def analyze_pull_ups(self, joints):
        """Enhanced pull-up analysis"""
        feedback, accuracy = [], 100
        
        try:
            ls = self.get_joint_coords(joints, 'LEFT_SHOULDER')
            lw = self.get_joint_coords(joints, 'LEFT_WRIST')
            rs = self.get_joint_coords(joints, 'RIGHT_SHOULDER')
            rw = self.get_joint_coords(joints, 'RIGHT_WRIST')
            
            if all(coord != [0, 0] for coord in [ls, lw, rs, rw]):
                # Use height difference between shoulder and wrist
                left_height_diff = ls[1] - lw[1]
                right_height_diff = rs[1] - rw[1]
                avg_height_diff = (left_height_diff + right_height_diff) / 2
                
                self.count_rep_with_stability(avg_height_diff, ExerciseType.PULL_UPS)
                
                params = self.exercise_params[ExerciseType.PULL_UPS]
                
                # Symmetry check
                asymmetry = abs(left_height_diff - right_height_diff)
                if asymmetry > params['max_asymmetry']:
                    feedback.append("Pull up evenly with both arms")
                    accuracy -= min(15, asymmetry * 0.5)
                
                # Height achievement
                if self.is_down_position:  # At the top
                    if avg_height_diff < params['optimal_range'][1]:
                        feedback.append("Pull higher - chin over bar")
                        accuracy -= 25
                
                # Full extension at bottom
                if not self.is_down_position:
                    if avg_height_diff > params['optimal_range'][0]:
                        feedback.append("Lower to full arm extension")
                        accuracy -= 12
                
                # Range of motion
                if self.range_of_motion < 40:
                    feedback.append("Increase range of motion")
                    accuracy -= 18
                elif self.range_of_motion > 70:
                    accuracy += 5
                    
            else:
                feedback.append("Position for frontal view")
                accuracy = 0
                
        except Exception:
            accuracy = 0
            
        return feedback, max(0, min(100, accuracy))

# Main application
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI Form Corrector Professional")
    app.setStyle('Fusion')  # Modern look
    
    # Set application icon
    app.setWindowIcon(app.style().standardIcon(QStyle.SP_ComputerIcon))
    
    window = AIFormCorrectorGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    print("🚀 Starting AI Form Corrector Professional Edition...")
    print("📋 Requirements: PyQt5, OpenCV, MediaPipe, NumPy")
    print("💡 New Features: Enhanced rep accuracy, session tracking, advanced analytics")
    print("=" * 70)
    
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install PyQt5 opencv-python mediapipe numpy")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()