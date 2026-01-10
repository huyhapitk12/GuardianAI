# Panel thư viện để xem lại các bản ghi


import cv2
import gc
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
import customtkinter as ctk
from customtkinter import CTkImage
from customtkinter import CTkScrollableFrame

from config import settings
from utils import security
from gui.styles import Colors, Fonts, Sizes, create_button, create_card


# Thư viện để xem ảnh và video
class GalleryPanel(ctk.CTkFrame):
    
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.playing = False
        self.temp_file: Optional[Path] = None
        self.stop_event = threading.Event()
        self.current_frame = 0
        self.image_ref = None
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        
        self.build_list_panel()
        self.build_preview_panel()
        self.refresh()
    
    # Xây dựng panel danh sách file
    def build_list_panel(self):
        panel = create_card(self)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, Sizes.SM))
        
        # Header
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        
        ctk.CTkLabel(header, text="Timeline", font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY).pack(side="left")
        create_button(header, "Refresh", "ghost", "small", width=60, command=self.refresh).pack(side="right")
        
        # List
        self.list_frame = CTkScrollableFrame(panel, fg_color="transparent")
        self.list_frame.pack(fill="both", expand=True)
    
    # Xây dựng panel xem trước
    def build_preview_panel(self):
        panel = create_card(self)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)
        
        # Info
        self.info_label = ctk.CTkLabel(panel, text="Select item", font=Fonts.TITLE_SM, text_color=Colors.TEXT_PRIMARY)
        self.info_label.grid(row=0, column=0, pady=Sizes.MD)
        
        # Preview area
        self.preview = ctk.CTkLabel(panel, text="", text_color=Colors.TEXT_MUTED)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=Sizes.MD)
        
        # Controls
        self.controls = ctk.CTkFrame(panel, fg_color="transparent")
        self.controls.grid(row=2, column=0, pady=Sizes.MD)
        
        self.play_btn = create_button(self.controls, "Play", "primary", "small", width=100, command=self.toggle_play)
        self.play_btn.pack()
        
        self.controls.grid_remove()
    
    # Làm mới danh sách file
    def refresh(self):
        # Clear list
        for widget in self.list_frame.winfo_children():
            widget.destroy()
        
        tmp_dir = settings.paths.tmp_dir
        if not tmp_dir.exists():
            return
        
        # Get files
        files = []
        for f in tmp_dir.iterdir():
            if f.suffix.lower() in ('.jpg', '.png', '.mp4'):
                files.append({
                    'path': f,
                    'time': f.stat().st_mtime,
                    'name': f.name,
                    'type': 'video' if f.suffix == '.mp4' else 'image'
                })
        
        files.sort(key=lambda x: x['time'], reverse=True)
        
        if not files:
            ctk.CTkLabel(self.list_frame, text="No recordings", text_color=Colors.TEXT_MUTED).pack(pady=Sizes.LG)
            return
        
        # Create buttons
        for item in files:
            dt = datetime.fromtimestamp(item['time'])
            icon = "[VID]" if item['type'] == 'video' else "[IMG]"
            
            btn = ctk.CTkButton(
                self.list_frame,
                text=f"{icon} {dt.strftime('%H:%M:%S')}\n{item['name']}",
                font=Fonts.BODY,
                fg_color=Colors.BG_TERTIARY,
                hover_color=Colors.BG_ELEVATED,
                height=50,
                anchor="w",
                command=lambda i=item: self.load(i)
            )
            btn.pack(fill="x", pady=2)
    
    # Tải mục được chọn
    def load(self, item: dict):
        self.cleanup()
        self.info_label.configure(text=item['name'])
        self.set_preview(text="Loading...")
        
        def load_thread():
            if item['type'] == 'image':
                self.after(0, lambda: self.controls.grid_remove())
                self.load_image(item['path'])
            else:
                self.after(0, lambda: self.controls.grid())
                self.after(0, lambda: self.play_btn.configure(text="Play"))
                self.load_video(item['path'])
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    # Tải file ảnh
    def load_image(self, path: Path):
        img = security.load_image(path)
        if img is None:
            self.set_preview(text="Failed to load image")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        def update_ui():
            if not self.winfo_exists():
                return
            
            w, h = self.preview.winfo_width(), self.preview.winfo_height()
            if w > 0 and h > 0:
                pil_img.thumbnail((w, h))
            
            ctk_img = CTkImage(pil_img, size=pil_img.size)
            self.image_ref = ctk_img
            self.set_preview(image=ctk_img, text="")
        
        self.after(0, update_ui)
    
    # Tải file video
    def load_video(self, path: Path):
        self.current_frame = 0
        
        # Decrypt to temp file
        data = security.decrypt_file(path)
        if not data:
            self.set_preview(text="Failed to decrypt video")
            return
        
        self.temp_file = settings.paths.tmp_dir / f"temp_{uuid.uuid4().hex}.mp4"
        self.temp_file.write_bytes(data)
        
        del data
        gc.collect()
        
        # Show first frame
        cap = cv2.VideoCapture(str(self.temp_file))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            def update_ui():
                if not self.winfo_exists():
                    return
                
                pil_img.thumbnail((640, 480))
                ctk_img = CTkImage(pil_img, size=pil_img.size)
                self.image_ref = ctk_img
                self.set_preview(image=ctk_img, text="")
            
            self.after(0, update_ui)
        else:
            self.set_preview(text="Failed to load video")
    
    # Bật/tắt phát video
    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()
    
    # Bắt đầu phát
    def play(self):
        if not self.temp_file or not self.temp_file.exists():
            return
        
        self.playing = True
        self.play_btn.configure(text="Pause")
        self.stop_event.clear()
        
        threading.Thread(target=self.playback_loop, daemon=True).start()
    
    # Tạm dừng phát
    def pause(self):
        self.playing = False
        self.stop_event.set()
    
    # Vòng lặp phát video
    def playback_loop(self):
        cap = cv2.VideoCapture(str(self.temp_file))
        
        if self.current_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        
        def update_frame():
            if not self.playing or self.stop_event.is_set():
                self.current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.release()
                self.play_btn.configure(text="Resume")
                return
            
            ret, frame = cap.read()
            if not ret:
                cap.release()
                self.playing = False
                self.current_frame = 0
                self.play_btn.configure(text="Play")
                return
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            w, h = self.preview.winfo_width(), self.preview.winfo_height()
            if w > 10 and h > 10:
                pil_img.thumbnail((w, h))
            
            ctk_img = CTkImage(pil_img, size=pil_img.size)
            self.image_ref = ctk_img
            self.set_preview(image=ctk_img)
            
            self.after(33, update_frame)
        
        self.after(0, update_frame)
    
    # Cập nhật preview an toàn
    def set_preview(self, **kwargs):
        try:
            if not self.winfo_exists() or not self.preview.winfo_exists():
                return
            self.preview.configure(**kwargs)
        except Exception:
            # Ignore TclError when image doesn't exist anymore
            pass
    
    # Dọn dẹp tài nguyên
    def cleanup(self):
        self.pause()
        self.current_frame = 0
        
        # Clear preview trước khi xóa image reference
        self.set_preview(image="", text="")
        self.image_ref = None
        
        if self.temp_file and self.temp_file.exists():
            try:
                self.temp_file.unlink()
            except Exception:
                pass
        self.temp_file = None
    
    def destroy(self):
        self.cleanup()
        super().destroy()