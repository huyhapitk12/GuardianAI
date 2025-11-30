import cv2
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Literal, Dict, List, Optional
from pathlib import Path
from PIL import Image

import customtkinter as ctk
from customtkinter import CTkFrame, CTkLabel, CTkScrollableFrame, CTkProgressBar, CTkCanvas, CTkButton, CTkImage, StringVar, CTkSwitch

from config import settings
from utils.security import security_manager
from .styles import Colors, Fonts, Sizes, create_card_frame, create_modern_button

# ============================================================================
# ACTIVITY LOGGER
# ============================================================================

ActivityType = Literal["info", "success", "warning", "error", "detection", "alert"]

class ActivityLogger:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
        self.activities = deque(maxlen=100)
        self.logs = deque(maxlen=200)
        self.activity_widgets = []
        self.log_widgets = []
        self._initialized = True
    
    def log_activity(self, message: str, activity_type: ActivityType = "info"):
        self.activities.append({'message': message, 'type': activity_type, 'timestamp': datetime.now()})
        for w in self.activity_widgets:
            try: w.add_activity(message, activity_type)
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                pass
            
    def log_system(self, message: str, level: Literal["info", "success", "warning", "error"] = "info"):
        self.logs.append({'message': message, 'level': level, 'timestamp': datetime.now()})
        for w in self.log_widgets:
            try: w.add_log(message, level)
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                pass

    def register_activity_widget(self, widget): 
        if widget not in self.activity_widgets: self.activity_widgets.append(widget)
    def register_log_widget(self, widget):
        if widget not in self.log_widgets: self.log_widgets.append(widget)
    def unregister_activity_widget(self, widget):
        if widget in self.activity_widgets: self.activity_widgets.remove(widget)
    def unregister_log_widget(self, widget):
        if widget in self.log_widgets: self.log_widgets.remove(widget)

activity_logger = ActivityLogger()
def log_activity(msg, type="info"): activity_logger.log_activity(msg, type)
def log_system(msg, level="info"): activity_logger.log_system(msg, level)

class ActivityFeedWidget(CTkScrollableFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=Colors.BG_PRIMARY, corner_radius=Sizes.CORNER_RADIUS, **kwargs)
        self.logger = ActivityLogger()
        self.logger.register_activity_widget(self)
        for act in list(self.logger.activities): self.add_activity(act['message'], act['type'])

    def add_activity(self, message: str, status: ActivityType = "info"):
        # Limit number of widgets to prevent lag
        MAX_ITEMS = 50
        children = self.winfo_children()
        if len(children) >= MAX_ITEMS:
            # Remove oldest items (first ones in pack order)
            for i in range(len(children) - MAX_ITEMS + 1):
                children[i].destroy()

        item = CTkFrame(self, fg_color=Colors.BG_TERTIARY, corner_radius=Sizes.CORNER_RADIUS_SM)
        item.pack(fill="x", pady=2)
        
        CTkLabel(item, text=datetime.now().strftime("%H:%M"), font=Fonts.TINY, text_color=Colors.TEXT_MUTED).pack(anchor="w", padx=Sizes.PADDING_SM, pady=(Sizes.PADDING_XS, 0))
        
        icons = {"success": "✅", "warning": "⚠️", "error": "❌", "info": "ℹ️", "detection": "🔍", "alert": "🚨"}
        colors = {"success": Colors.SUCCESS, "warning": Colors.WARNING, "error": Colors.DANGER, "info": Colors.INFO, "detection": Colors.PRIMARY, "alert": Colors.DANGER}
        
        CTkLabel(item, text=f"{icons.get(status, 'ℹ️')} {message}", font=Fonts.CAPTION, text_color=colors.get(status, Colors.TEXT_SECONDARY), wraplength=350).pack(anchor="w", padx=Sizes.PADDING_SM, pady=(0, Sizes.PADDING_XS))

    def destroy(self):
        self.logger.unregister_activity_widget(self)
        super().destroy()

class SystemLogsWidget(CTkScrollableFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=Colors.BG_PRIMARY, corner_radius=Sizes.CORNER_RADIUS, **kwargs)
        self.logger = ActivityLogger()
        self.logger.register_log_widget(self)
        for log in list(self.logger.logs): self.add_log(log['message'], log['level'])

    def add_log(self, message: str, level="info"):
        # Limit number of widgets
        MAX_ITEMS = 100
        children = self.winfo_children()
        if len(children) >= MAX_ITEMS:
            for i in range(len(children) - MAX_ITEMS + 1):
                children[i].destroy()

        colors = {"info": Colors.TEXT_SECONDARY, "success": Colors.SUCCESS, "warning": Colors.WARNING, "error": Colors.DANGER}
        frame = CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", pady=1)
        CTkLabel(frame, text=f"[{datetime.now().strftime('%H:%M:%S')}]", font=Fonts.TINY, text_color=Colors.TEXT_MUTED, width=80).pack(side="left")
        CTkLabel(frame, text=message, font=Fonts.TINY, text_color=colors.get(level, Colors.TEXT_SECONDARY), wraplength=350).pack(side="left", padx=Sizes.PADDING_SM)

    def destroy(self):
        self.logger.unregister_log_widget(self)
        super().destroy()

# ============================================================================
# MONITORING WIDGETS
# ============================================================================

class CameraHealthWidget(CTkFrame):
    def __init__(self, parent, camera_manager, **kwargs):
        super().__init__(parent, fg_color="transparent", height=200, **kwargs)
        self.camera_manager = camera_manager
        self.widgets = {}
        
        # Header
        h = CTkFrame(self, fg_color="transparent")
        h.pack(fill="x", padx=Sizes.PADDING_SM, pady=(Sizes.PADDING_SM, 0))
        
        tf = CTkFrame(h, fg_color="transparent")
        tf.pack(side="left")
        CTkLabel(tf, text="📹", font=("Segoe UI", 20), text_color=Colors.PRIMARY).pack(side="left", padx=(0, Sizes.PADDING_SM))
        CTkLabel(tf, text="Camera Health", font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        
        sf = CTkFrame(h, fg_color="transparent")
        sf.pack(side="right")
        self.summary = CTkLabel(sf, text="Checking...", font=Fonts.SMALL, text_color=Colors.TEXT_MUTED)
        self.summary.pack(side="left")

        self.scroll = CTkScrollableFrame(self, fg_color=Colors.BG_PRIMARY, corner_radius=Sizes.CORNER_RADIUS)
        self.scroll.pack(fill="both", expand=True, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        
        if self.camera_manager:
            for i, (src, cam) in enumerate(self.camera_manager.cameras.items()):
                w = self._create_widget(src)
                w.pack(fill="x", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_XS)
                self.widgets[src] = w
        
        self._monitor()

    def _create_widget(self, source):
        c = create_card_frame(self.scroll, fg_color=Colors.BG_SECONDARY)
        c.lbls = {}
        
        h = CTkFrame(c, fg_color="transparent")
        h.pack(fill="x", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        CTkLabel(h, text=f"📹 Cam {source}", font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        
        c.lbls['dot'] = CTkFrame(h, width=12, height=12, corner_radius=6)
        c.lbls['dot'].pack(side="right", padx=(Sizes.PADDING_SM, 0))
        c.lbls['status'] = CTkLabel(h, text="...", font=Fonts.SMALL)
        c.lbls['status'].pack(side="right")
        
        s = CTkFrame(c, fg_color="transparent")
        s.pack(fill="x", padx=Sizes.PADDING_MD)
        c.lbls['fps'] = CTkLabel(s, text="FPS: 0", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        c.lbls['fps'].pack(side="left", padx=(0, 10))
        c.lbls['res'] = CTkLabel(s, text="Res: 0x0", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        c.lbls['res'].pack(side="left")
        
        c.prog = CTkProgressBar(c, height=5, progress_color=Colors.SUCCESS)
        c.prog.pack(fill="x", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_SM)
        
        b = CTkFrame(c, fg_color="transparent")
        b.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        create_modern_button(b, "Reconnect", "secondary", "small", command=lambda: self._reconnect(source)).pack(side="left")
        return c

    def _reconnect(self, src):
        cam = self.camera_manager.get_camera(src)
        if cam: cam.force_reconnect()

    def _monitor(self):
        try:
            healthy = 0
            total = len(self.camera_manager.cameras) if self.camera_manager else 0
            
            for src, w in self.widgets.items():
                cam = self.camera_manager.get_camera(src)
                if not cam: continue
                stat = cam.get_connection_status()
                
                if isinstance(stat, dict):
                    is_healthy = stat.get('is_healthy', False)
                else:
                    is_healthy = bool(stat)
                
                if is_healthy:
                    w.lbls['status'].configure(text="Connected", text_color=Colors.SUCCESS)
                    w.lbls['dot'].configure(fg_color=Colors.SUCCESS)
                    w.prog.set(1.0)
                    healthy += 1
                else:
                    w.lbls['status'].configure(text="Offline", text_color=Colors.DANGER)
                    w.lbls['dot'].configure(fg_color=Colors.DANGER)
                    w.prog.set(0.0)
                
                if hasattr(cam, 'cap') and cam.cap:
                    fps = int(cam.cap.get(cv2.CAP_PROP_FPS))
                    w.lbls['fps'].configure(text=f"FPS: {fps}")
                    w.lbls['res'].configure(text=f"Res: {int(cam.cap.get(3))}x{int(cam.cap.get(4))}")

            if total > 0:
                self.summary.configure(text=f"{healthy}/{total} Healthy", text_color=Colors.SUCCESS if healthy==total else Colors.WARNING)
        except Exception as e:
            print(f"Error in monitor loop: {e}")
            pass
        self.after(2000, self._monitor)

class FireHistoryWidget(CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.events = deque(maxlen=100)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        h = CTkFrame(self, fg_color="transparent")
        h.grid(row=0, column=0, sticky="ew", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        CTkLabel(h, text="🔥 Fire History", font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        self.stats = CTkLabel(h, text="No events", font=Fonts.SMALL, text_color=Colors.TEXT_SECONDARY)
        self.stats.pack(side="right")
        
        self.canvas = CTkCanvas(self, bg=Colors.BG_SECONDARY, height=100, highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="ew", padx=Sizes.PADDING_SM)
        
        self.scroll = CTkScrollableFrame(self, fg_color=Colors.BG_PRIMARY, corner_radius=Sizes.CORNER_RADIUS)
        self.scroll.grid(row=2, column=0, sticky="nsew", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        
        self._update()

    def log_fire_event(self, type, source, area, conf):
        self.events.append({'timestamp': datetime.now(), 'type': type, 'source': source, 'area': area, 'conf': conf})
        self._update()

    def _update(self):
        # Stats
        if self.events:
            recent = sum(1 for e in self.events if (datetime.now()-e['timestamp']).total_seconds() < 3600)
            self.stats.configure(text=f"Total: {len(self.events)} | Last hr: {recent}")
        
        # List
        for w in self.scroll.winfo_children(): w.destroy()
        if not self.events:
            CTkLabel(self.scroll, text="No fire events", text_color=Colors.TEXT_MUTED).pack(pady=20)
        else:
            for e in reversed(list(self.events)[-20:]):
                f = CTkFrame(self.scroll, fg_color=Colors.BG_TERTIARY)
                f.pack(fill="x", pady=2)
                CTkLabel(f, text=e['timestamp'].strftime("%H:%M:%S"), font=Fonts.SMALL, text_color=Colors.TEXT_SECONDARY).pack(side="left", padx=5)
                c = Colors.DANGER if e['type']=='fire' else Colors.WARNING
                CTkLabel(f, text=f"{'🔥' if e['type']=='fire' else '💨'} {e['type'].upper()}", font=Fonts.SMALL, text_color=c).pack(side="left", padx=5)
                CTkLabel(f, text=f"{int(e['conf']*100)}%", font=Fonts.SMALL, text_color=Colors.SUCCESS if e['conf']>0.9 else Colors.WARNING).pack(side="right", padx=5)

# ============================================================================
# GALLERY
# ============================================================================

class GalleryPanel(CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        self.playing = False
        self.temp_vid = None
        self.stop_evt = threading.Event()
        self.current_image = None
        self.current_frame = 0  # Track video position

        # List
        lp = CTkFrame(self, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.CORNER_RADIUS)
        lp.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.PADDING_SM))
        h = CTkFrame(lp, fg_color="transparent")
        h.pack(fill="x", padx=Sizes.PADDING_MD, pady=Sizes.PADDING_MD)
        CTkLabel(h, text="Timeline", font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        CTkButton(h, text="🔄", width=30, command=self.refresh).pack(side="right")
        self.list = CTkScrollableFrame(lp, fg_color="transparent")
        self.list.pack(fill="both", expand=True)
        
        # Preview
        pp = CTkFrame(self, fg_color=Colors.BG_SECONDARY, corner_radius=Sizes.CORNER_RADIUS)
        pp.grid(row=0, column=1, sticky="nsew")
        pp.grid_columnconfigure(0, weight=1)
        pp.grid_rowconfigure(1, weight=1)
        
        self.info = CTkLabel(pp, text="Select item", font=Fonts.TITLE_SM, text_color=Colors.TEXT_PRIMARY)
        self.info.grid(row=0, column=0, pady=Sizes.PADDING_MD)
        self.view = CTkLabel(pp, text="", text_color=Colors.TEXT_MUTED)
        self.view.grid(row=1, column=0, sticky="nsew", padx=Sizes.PADDING_MD)
        
        self.ctrl = CTkFrame(pp, fg_color="transparent")
        self.ctrl.grid(row=2, column=0, pady=Sizes.PADDING_MD)
        self.play_btn = CTkButton(self.ctrl, text="▶ Play", width=100, command=self.toggle_play)
        self.play_btn.pack()
        self.ctrl.grid_remove()
        
        self.refresh()

    def refresh(self):
        for w in self.list.winfo_children(): w.destroy()
        tmp = settings.paths.tmp_dir
        if not tmp.exists(): return
        
        files = []
        for f in tmp.iterdir():
            if f.suffix.lower() in ['.jpg', '.png', '.mp4']:
                try: files.append({'path': f, 'time': f.stat().st_mtime, 'name': f.name, 'type': 'video' if f.suffix=='.mp4' else 'image'})
                except Exception as e:
                    print(f"Error in monitor loop: {e}")
                    pass
        
        files.sort(key=lambda x: x['time'], reverse=True)
        if not files: CTkLabel(self.list, text="No recordings").pack(pady=20)
        else:
            for i in files:
                dt = datetime.fromtimestamp(i['time'])
                icon = "🎥" if i['type']=='video' else "📸"
                btn = CTkButton(self.list, text=f"{icon} {dt.strftime('%H:%M:%S')}\n{i['name']}", 
                                font=Fonts.BODY, fg_color=Colors.BG_TERTIARY, hover_color=Colors.BG_ELEVATED, 
                                height=50, anchor="w", command=lambda x=i: self.load(x))
                btn.pack(fill="x", pady=2)

    def _safe_update_view(self, **kwargs):
        """Safely update view, recreating widget if TclError occurs"""
        try:
            self.view.configure(**kwargs)
        except Exception:
            try:
                # Recreate widget
                parent = self.view.master
                self.view.destroy()
                self.view = CTkLabel(parent, text="", text_color=Colors.TEXT_MUTED)
                self.view.grid(row=1, column=0, sticky="nsew", padx=Sizes.PADDING_MD)
                self.view.configure(**kwargs)
            except Exception:
                pass

    def load(self, item):
        self.close_video()
        self.info.configure(text=item['name'])
        # Clean up any existing image reference when loading new item
        if hasattr(self.view, '_image_ref'):
            delattr(self.view, '_image_ref')
            
        self._safe_update_view(image=None, text="Loading...")

        def load_media_thread():
            try:
                if item['type'] == 'image':
                    # Update UI from thread
                    self.root_after_safe(lambda: self.ctrl.grid_remove())
                    self._show_img(item['path'])
                else:
                    self.root_after_safe(lambda: self.ctrl.grid())
                    self.root_after_safe(lambda: self.play_btn.configure(text="▶ Play", state="normal"))
                    self._prep_vid(item['path'])
            except Exception as e:
                print(f"Error loading media: {e}")
                self._safe_update_view(text="Error loading file")

        threading.Thread(target=load_media_thread, daemon=True).start()

    def root_after_safe(self, func):
        """Helper to schedule on main thread safely"""
        try:
            self.after(0, func)
        except Exception:
            pass

    def _show_img(self, path):
        try:
            img = security_manager.decrypt_image(path)
            if img is None: 
                self._safe_update_view(text="Decryption failed", image=None)
                return
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
            
            # Get size safely on main thread
            # We can't access winfo_width directly from thread safely sometimes, 
            # but usually reading is fine. Writing is the issue.
            # To be safe, we'll just process the image and schedule update.
            
            def update_ui_with_image():
                if not self.winfo_exists(): return
                w, h = self.view.winfo_width(), self.view.winfo_height()
                if w>0 and h>0: pil.thumbnail((w, h))
                ctk_img = CTkImage(pil, size=pil.size)
                
                # Store reference to prevent garbage collection
                setattr(self.view, '_image_ref', ctk_img)
                self._safe_update_view(image=ctk_img, text="")
            
            self.root_after_safe(update_ui_with_image)
            
        except Exception as e:
            print(f"Image load error: {e}")
            self._safe_update_view(text="Error loading image")

    def _prep_vid(self, path):
        self._safe_update_view(image=None, text="Loading video...")
        self.current_frame = 0 # Reset position for new video
        try:
            data = security_manager.try_decrypt_file(path)
            if not data: 
                self._safe_update_view(text="Decryption failed")
                return
            
            self.temp_vid = settings.paths.tmp_dir / f"temp_{uuid.uuid4().hex}.mp4"
            with open(self.temp_vid, 'wb') as f: f.write(data)
            
            # Explicitly free memory
            del data
            import gc
            gc.collect()
            
            cap = cv2.VideoCapture(str(self.temp_vid))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame)
                
                def update_ui_with_video_thumb():
                    if not self.winfo_exists(): return
                    pil.thumbnail((640, 480))
                    ctk_img = CTkImage(pil, size=pil.size)
                    
                    setattr(self.view, '_image_ref', ctk_img)
                    self._safe_update_view(image=ctk_img, text="")
                
                self.root_after_safe(update_ui_with_video_thumb)
            else:
                self._safe_update_view(text="Failed to load video frame")
        except Exception as e: 
            print(f"Video prep error: {e}")
            try:
                self._safe_update_view(text=f"Error loading video", image=None)
            except:
                pass

    def toggle_play(self):
        if self.playing: self.stop()
        else:
            if self.temp_vid and self.temp_vid.exists():
                self.playing = True
                self.play_btn.configure(text="⏸ Pause")
                self.stop_evt.clear()
                threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        cap = cv2.VideoCapture(str(self.temp_vid))
        
        # Resume from last position
        if self.current_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        def update_frame():
            if not self.playing or self.stop_evt.is_set():
                # Save position before releasing
                self.current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.release()
                try: self.play_btn.configure(text="▶ Resume")
                except Exception as e:
                    print(f"Error in monitor loop: {e}")
                    pass
                return

            ret, frame = cap.read()
            if not ret:
                cap.release()
                self.playing = False
                self.current_frame = 0 # Reset when finished
                try: self.play_btn.configure(text="▶ Play")
                except Exception as e:
                    print(f"Error in monitor loop: {e}")
                    pass
                return
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame)
            w, h = self.view.winfo_width(), self.view.winfo_height()
            if w>10 and h>10: pil.thumbnail((w, h))
            
            try: 
                ctk_img = CTkImage(pil, size=pil.size)
                self._safe_update_view(image=ctk_img)
                setattr(self.view, '_image_ref', ctk_img)
            except: 
                cap.release()
                return

            self.after(33, update_frame)

        self.after(0, update_frame)

    def stop(self):
        """Pause playback"""
        self.playing = False
        self.stop_evt.set()

    def close_video(self):
        """Stop playback and clean up resources"""
        self.stop()
        self.current_frame = 0 # Reset position
        
        self._safe_update_view(image=None)

        if self.temp_vid and self.temp_vid.exists():
            try: self.temp_vid.unlink()
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                pass
        self.temp_vid = None
        
        if hasattr(self.view, '_image_ref'):
            delattr(self.view, '_image_ref')

    def destroy(self):
        self.close_video()
        super().destroy()



class UnifiedCameraList(CTkFrame):
    """Unified widget showing both controls and health for each camera"""
    def __init__(self, parent, camera_manager, state_manager, on_view_command=None, on_add_command=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.camera_manager = camera_manager
        self.state = state_manager
        self.on_view_command = on_view_command
        self.on_add_command = on_add_command
        self.widgets = {}
        self.switches = {}
        
        # Header
        h = CTkFrame(self, fg_color="transparent")
        h.pack(fill="x", padx=Sizes.PADDING_SM, pady=(Sizes.PADDING_SM, 0))
        
        tf = CTkFrame(h, fg_color="transparent")
        tf.pack(side="left")
        CTkLabel(tf, text="🎥", font=("Segoe UI", 20), text_color=Colors.PRIMARY).pack(side="left", padx=(0, Sizes.PADDING_SM))
        CTkLabel(tf, text="Camera Control", font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        
        # Add Camera Button
        if self.on_add_command:
            create_modern_button(h, "+ Add", "primary", "small", width=60, command=self.on_add_command).pack(side="right")

        self.scroll = CTkScrollableFrame(self, fg_color=Colors.BG_PRIMARY, corner_radius=Sizes.CORNER_RADIUS)
        self.scroll.pack(fill="both", expand=True, padx=Sizes.PADDING_SM, pady=Sizes.PADDING_SM)
        
        if self.camera_manager:
            for i, (src, cam) in enumerate(self.camera_manager.cameras.items()):
                w = self._create_widget(src)
                w.pack(fill="x", padx=Sizes.PADDING_SM, pady=Sizes.PADDING_XS)
                self.widgets[src] = w
        
        self._monitor()

    def _create_widget(self, source):
        c = create_card_frame(self.scroll, fg_color=Colors.BG_SECONDARY)
        c.lbls = {}
        
        # Top Row: Name + Status + Switch
        h = CTkFrame(c, fg_color="transparent")
        h.pack(fill="x", padx=Sizes.PADDING_MD, pady=(Sizes.PADDING_MD, 5))
        
        # Name
        CTkLabel(h, text=f"📹 Cam {source}", font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        
        # Switch (Right aligned)
        switch_var = StringVar(value="on" if self.state.is_person_detection_enabled(source) else "off")
        switch = CTkSwitch(
            h,
            text="Detect",
            variable=switch_var,
            onvalue="on",
            offvalue="off",
            command=lambda s=source, v=switch_var: self._toggle_cam(s, v),
            width=80,
            height=24,
            font=Fonts.SMALL,
            progress_color=Colors.SUCCESS
        )
        switch.pack(side="right")
        self.switches[source] = switch_var
        
        # Status Dot (Right of Name)
        c.lbls['dot'] = CTkFrame(h, width=10, height=10, corner_radius=5)
        c.lbls['dot'].pack(side="right", padx=Sizes.PADDING_MD)

        # Middle Row: Stats
        s = CTkFrame(c, fg_color="transparent")
        s.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, 5))
        
        c.lbls['fps'] = CTkLabel(s, text="FPS: 0", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        c.lbls['fps'].pack(side="left", padx=(0, 10))
        c.lbls['res'] = CTkLabel(s, text="Res: 0x0", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        c.lbls['res'].pack(side="left")
        
        c.lbls['status'] = CTkLabel(s, text="...", font=Fonts.TINY)
        c.lbls['status'].pack(side="right")

        # IR Status Row
        ir_frame = CTkFrame(c, fg_color="transparent")
        ir_frame.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, 5))
        
        c.lbls['ir_status'] = CTkLabel(ir_frame, text="IR: OFF", font=Fonts.TINY, text_color=Colors.TEXT_MUTED)
        c.lbls['ir_status'].pack(side="left")
        
        # IR Enhance Switch
        ir_enhance_var = StringVar(value="on" if getattr(self.camera_manager.get_camera(source), 'ir_enhancement_enabled', False) else "off")
        ir_switch = CTkSwitch(
            ir_frame,
            text="IR Enhance",
            variable=ir_enhance_var,
            onvalue="on",
            offvalue="off",
            command=lambda s=source, v=ir_enhance_var: self._toggle_ir_enhance(s, v),
            width=80,
            height=20,
            font=Fonts.TINY,
            progress_color=Colors.WARNING
        )
        ir_switch.pack(side="right")
        self.switches[f"{source}_ir"] = ir_enhance_var

        # Bottom Row: Actions (View + Reconnect)
        b = CTkFrame(c, fg_color="transparent")
        b.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_MD))
        
        # View Button
        if self.on_view_command:
            create_modern_button(b, "View", "primary", "small", width=60, command=lambda s=source: self.on_view_command(s)).pack(side="left", padx=(0, 5))

        # Reconnect Button
        create_modern_button(b, "⟳", "secondary", "small", width=30, command=lambda: self._reconnect(source)).pack(side="right")
        
        # Progress Bar (at very bottom)
        c.prog = CTkProgressBar(c, height=4, progress_color=Colors.SUCCESS)
        c.prog.pack(fill="x", padx=Sizes.PADDING_MD, pady=(0, Sizes.PADDING_SM))
        
        return c

    def _toggle_cam(self, source, var):
        enabled = (var.get() == "on")
        self.state.set_person_detection(enabled, source)
        if enabled:
            log_activity(f"Enabled detection for Camera {source}", "success")
        else:
            log_activity(f"Disabled detection for Camera {source}", "warning")

    def _reconnect(self, src):
        cam = self.camera_manager.get_camera(src)
        if cam: cam.force_reconnect()

    def _toggle_ir_enhance(self, source, var):
        cam = self.camera_manager.get_camera(source)
        if cam and hasattr(cam, 'set_ir_enhancement'):
            enabled = (var.get() == "on")
            cam.set_ir_enhancement(enabled)

    def _monitor(self):
        try:
            # Removed summary label update as it was removed from header
            for src, w in self.widgets.items():
                cam = self.camera_manager.get_camera(src)
                if not cam: continue
                stat = cam.get_connection_status()
                
                if isinstance(stat, dict):
                    is_healthy = stat.get('is_healthy', False)
                else:
                    is_healthy = bool(stat)
                
                if is_healthy:
                    w.lbls['status'].configure(text="Online", text_color=Colors.SUCCESS)
                    w.lbls['dot'].configure(fg_color=Colors.SUCCESS)
                    w.prog.set(1.0)
                else:
                    w.lbls['status'].configure(text="Offline", text_color=Colors.DANGER)
                    w.lbls['dot'].configure(fg_color=Colors.DANGER)
                    w.prog.set(0.0)
                
                if hasattr(cam, 'cap') and cam.cap:
                    fps = int(cam.cap.get(cv2.CAP_PROP_FPS))
                    w.lbls['fps'].configure(text=f"FPS: {fps}")
                    w.lbls['res'].configure(text=f"{int(cam.cap.get(3))}x{int(cam.cap.get(4))}")
                
                # Update IR Status
                if hasattr(cam, 'get_infrared_status'):
                    is_ir = cam.get_infrared_status()
                    w.lbls['ir_status'].configure(
                        text="IR: ON" if is_ir else "IR: OFF",
                        text_color=Colors.WARNING if is_ir else Colors.TEXT_MUTED
                    )
        except Exception as e:
            print(f"Error in monitor loop: {e}")
            pass
        self.after(2000, self._monitor)
