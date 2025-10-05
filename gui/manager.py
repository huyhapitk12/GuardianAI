"""GUI for face management"""
import logging
import os
import shutil
import customtkinter as ctk
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path
from PIL import Image
import cv2

from config.settings import settings

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FaceManagerGUI:
    """Face management GUI application"""
    
    def __init__(self, root, camera, face_detector, state_manager):
        self.root = root
        self.camera = camera
        self.face_detector = face_detector
        self.state = state_manager
        
        self.current_person = None
        self.thumbnail_size = (100, 100)
        self.main_image_size = (350, 350)
        self.video_feed_size = (640, 480)
        
        self._setup_window()
        self._create_sidebar()
        self._create_content_area()
        
        self.populate_person_list()
        self.show_video_view()
        self._start_video_feed()
        self._sync_detection_switch()
    
    def _setup_window(self):
        """Setup main window"""
        self.root.title("Guardian - Face Manager")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
    
    def _create_sidebar(self):
        """Create left sidebar"""
        self.left_frame = ctk.CTkFrame(self.root, width=250, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsw")
        self.left_frame.grid_rowconfigure(2, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            self.left_frame,
            text="Điều khiển",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Show camera button
        self.btn_show_cam = ctk.CTkButton(
            self.left_frame,
            text="Xem Camera Trực Tiếp",
            command=self.show_video_view
        )
        self.btn_show_cam.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # Person list
        self.person_list_frame = ctk.CTkScrollableFrame(
            self.left_frame,
            label_text="Người đã lưu"
        )
        self.person_list_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Control panel
        self._create_control_panel()
    
    def _create_control_panel(self):
        """Create control panel with settings"""
        panel = ctk.CTkFrame(self.left_frame)
        panel.grid(row=3, column=0, padx=10, pady=10, sticky="sew")
        
        # Detection toggle
        detection_frame = ctk.CTkFrame(panel)
        detection_frame.pack(pady=10, padx=10, fill="x")
        
        self.detection_switch_var = ctk.StringVar(
            value="on" if self.state.is_person_detection_enabled() else "off"
        )
        self.detection_switch = ctk.CTkSwitch(
            detection_frame,
            text="Nhận diện người",
            command=self._toggle_detection,
            variable=self.detection_switch_var,
            onvalue="on",
            offvalue="off"
        )
        self.detection_switch.pack(side="left", padx=10, pady=5)
        
        # YOLO model selector
        ctk.CTkLabel(panel, text="Model YOLO").pack(pady=(10, 0), padx=10, fill="x")
        self.yolo_var = ctk.StringVar(value=settings.models.yolo_size)
        yolo_combo = ctk.CTkOptionMenu(
            panel,
            variable=self.yolo_var,
            values=["medium", "small"],
            command=self._change_yolo_model
        )
        yolo_combo.pack(pady=5, padx=10, fill="x")
        
        # Face model selector
        ctk.CTkLabel(panel, text="Model Nhận diện mặt").pack(pady=(10, 0), padx=10, fill="x")
        self.face_model_var = ctk.StringVar(value=settings.models.face_model_name)
        face_combo = ctk.CTkOptionMenu(
            panel,
            variable=self.face_model_var,
            values=["buffalo_s", "buffalo_l"],
            command=self._change_face_model
        )
        face_combo.pack(pady=5, padx=10, fill="x")
        
        # Action buttons
        ctk.CTkButton(
            panel,
            text="Thêm Người Mới",
            command=self.add_person
        ).pack(pady=5, padx=10, fill="x")
        
        ctk.CTkButton(
            panel,
            text="Xây Dựng Lại Tất Cả",
            command=self.rebuild_all
        ).pack(pady=5, padx=10, fill="x")
        
        ctk.CTkButton(
            panel,
            text="Xóa Tất Cả",
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            command=self.delete_all
        ).pack(pady=(5, 10), padx=10, fill="x")
    
    def _create_content_area(self):
        """Create right content area"""
        self.right_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)
        
        # Video view
        self.video_view_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.video_label = ctk.CTkLabel(self.video_view_frame, text="Đang tải camera...")
        self.video_label.pack(expand=True, fill="both")
        
        # Person view
        self.person_view_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.person_view_frame.grid_columnconfigure(0, weight=1)
        self.person_view_frame.grid_rowconfigure(1, weight=1)
    
    def show_video_view(self):
        """Show camera video feed"""
        self.person_view_frame.grid_forget()
        self.video_view_frame.grid(row=0, column=0, sticky="nsew")
        self.current_person = None
        self.btn_show_cam.configure(state="disabled", fg_color=("gray75", "gray25"))
    
    def show_person_view(self):
        """Show person details view"""
        self.video_view_frame.grid_forget()
        self.person_view_frame.grid(row=0, column=0, sticky="nsew")
        self.btn_show_cam.configure(state="normal")
    
    def _start_video_feed(self):
        """Start video feed update loop"""
        def update():
            ret, frame = self.camera.read()
            if ret:
                self.video_label.configure(text="")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                ctk_image = ctk.CTkImage(pil_image, size=self.video_feed_size)
                
                if self.video_label.winfo_exists():
                    self.video_label.configure(image=ctk_image)
                    self.video_label.image = ctk_image
            
            self.root.after(15, update)
        
        update()
    
    def _sync_detection_switch(self):
        """Sync detection switch with state"""
        is_enabled = self.state.is_person_detection_enabled()
        state_str = "on" if is_enabled else "off"
        
        if self.detection_switch_var.get() != state_str:
            self.detection_switch_var.set(state_str)
        
        self.root.after(1000, self._sync_detection_switch)
    
    def _toggle_detection(self):
        """Toggle person detection"""
        is_on = self.detection_switch_var.get() == "on"
        self.state.set_person_detection_enabled(is_on)
    
    def _change_yolo_model(self, size: str):
        """Change YOLO model size"""
        try:
            # Update fire detector
            from core.detection.fire_detector import FireDetector
            fire_detector = FireDetector()
            fire_detector.update_model(size)
            
            # Update person tracker
            from core.detection.person_tracker import PersonTracker
            person_tracker = PersonTracker(self.face_detector)
            person_tracker.initialize()
            
            messagebox.showinfo("Thành công", f"Đã chuyển sang model YOLO: '{size}'")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể chuyển model: {e}")
    
    def _change_face_model(self, model_name: str):
        """Change face recognition model"""
        try:
            self.face_detector.update_model(model_name)
            messagebox.showinfo("Thành công", f"Đã chuyển sang model: {model_name}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể chuyển model: {e}")
    
    def populate_person_list(self):
        """Populate person list from data directory"""
        for widget in self.person_list_frame.winfo_children():
            widget.destroy()
        
        data_dir = settings.paths.data_dir
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            return
        
        persons = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        
        for person_name in persons:
            btn = ctk.CTkButton(
                self.person_list_frame,
                text=person_name,
                command=lambda name=person_name: self.select_person(name),
                fg_color="transparent",
                anchor="w",
                hover_color=("gray85", "gray20")
            )
            btn.pack(fill="x", padx=5, pady=2)
    
    def select_person(self, name: str):
        """Select and display a person's details"""
        self.current_person = name
        self.show_person_view()
        
        for widget in self.person_view_frame.winfo_children():
            widget.destroy()
        
        # Header
        header = ctk.CTkFrame(self.person_view_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            header,
            text=name,
            font=ctk.CTkFont(size=28, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=(0, 20))
        
        # Action buttons
        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.grid(row=0, column=2, sticky="e")
        
        ctk.CTkButton(
            actions,
            text="Thêm Ảnh",
            command=lambda: self.add_image_for_person(name)
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            actions,
            text="Xóa Người Này",
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            command=lambda: self.delete_person(name)
        ).pack(side="left", padx=5)
        
        # Main image
        self.main_image_label = ctk.CTkLabel(self.person_view_frame, text="")
        self.main_image_label.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Thumbnails
        self.thumbnail_frame = ctk.CTkScrollableFrame(
            self.person_view_frame,
            label_text="Ảnh đã lưu",
            height=120,
            orientation="horizontal"
        )
        self.thumbnail_frame.grid(row=2, column=0, sticky="ew", pady=10)
        
        self._load_person_images(name)
    
    def _load_person_images(self, name: str):
        """Load images for a person"""
        person_dir = settings.paths.data_dir / name
        image_files = [
            f for f in person_dir.iterdir()
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
        ]
        
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()
        
        if not image_files:
            self.main_image_label.configure(image=None, text="Không có ảnh nào")
            return
        
        # Display first image as main
        self._update_main_image(image_files[0])
        
        # Create thumbnails
        for img_file in image_files:
            try:
                img = Image.open(img_file)
                img.thumbnail(self.thumbnail_size)
                ctk_img = ctk.CTkImage(img, size=self.thumbnail_size)
                
                btn = ctk.CTkButton(
                    self.thumbnail_frame,
                    text="",
                    image=ctk_img,
                    width=self.thumbnail_size[0],
                    height=self.thumbnail_size[1],
                    command=lambda p=img_file: self._update_main_image(p)
                )
                btn.pack(side="left", padx=5, pady=5)
            except Exception as e:
                logger.error(f"Failed to create thumbnail: {e}")
    
    def _update_main_image(self, image_path: Path):
        """Update main image display"""
        try:
            img = Image.open(image_path)
            ctk_img = ctk.CTkImage(img, size=self.main_image_size)
            self.main_image_label.configure(image=ctk_img, text="")
            self.main_image_label.image = ctk_img
        except Exception as e:
            logger.error(f"Failed to display image: {e}")
            self.main_image_label.configure(image=None, text="Lỗi khi tải ảnh")
    
    def add_person(self):
        """Add a new person"""
        name = simpledialog.askstring(
            "Thêm Người Mới",
            "Nhập tên người cần thêm:",
            parent=self.root
        )
        
        if not name or not name.strip():
            if name is not None:
                messagebox.showwarning("Cảnh báo", "Tên không được để trống")
            return
        
        name = name.strip()
        person_dir = settings.paths.data_dir / name
        
        if person_dir.exists():
            messagebox.showerror("Lỗi", f"Người có tên '{name}' đã tồn tại")
            return
        
        img_path = filedialog.askopenfilename(
            title=f"Chọn ảnh đầu tiên cho {name}",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        
        if not img_path:
            return
        
        person_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, person_dir)
        
        # Rebuild embeddings
        self.face_detector.rebuild_embeddings()
        self.populate_person_list()
        self.select_person(name)
        
        messagebox.showinfo("Thành công", f"Đã thêm '{name}' thành công")
    
    def add_image_for_person(self, name: str):
        """Add image for existing person"""
        img_path = filedialog.askopenfilename(
            title=f"Chọn ảnh mới cho {name}",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        
        if not img_path:
            return
        
        person_dir = settings.paths.data_dir / name
        shutil.copy(img_path, person_dir)
        
        self.face_detector.rebuild_embeddings()
        self._load_person_images(name)
        
        messagebox.showinfo("Thành công", f"Đã thêm ảnh mới cho '{name}'")
    
    def delete_person(self, name: str):
        """Delete a person"""
        if not messagebox.askyesno(
            "Xác nhận",
            f"Bạn có chắc chắn muốn xóa tất cả dữ liệu của '{name}'?"
        ):
            return
        
        try:
            person_dir = settings.paths.data_dir / name
            shutil.rmtree(person_dir)
            
            self.face_detector.rebuild_embeddings()
            self.populate_person_list()
            self.show_video_view()
            
            messagebox.showinfo("Thành công", f"Đã xóa '{name}'")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi xóa: {e}")
    
    def rebuild_all(self):
        """Rebuild all face embeddings"""
        if not messagebox.askyesno(
            "Xác nhận",
            "Quá trình này sẽ mã hóa lại tất cả các khuôn mặt. "
            "Điều này có thể mất thời gian. Bạn có muốn tiếp tục?"
        ):
            return
        
        count = self.face_detector.rebuild_embeddings()
        messagebox.showinfo(
            "Hoàn tất",
            f"Đã xây dựng lại dữ liệu thành công.\n"
            f"Tổng số {count} khuôn mặt đã được mã hóa."
        )
        
        if self.current_person:
            self.select_person(self.current_person)
    
    def delete_all(self):
        """Delete all face data"""
        confirm1 = messagebox.askyesno(
            "CẢNH BÁO",
            "Bạn có chắc chắn muốn XÓA TẤT CẢ dữ liệu khuôn mặt không? "
            "Hành động này KHÔNG THỂ hoàn tác.",
            icon='warning'
        )
        if not confirm1:
            return
        
        confirm2 = simpledialog.askstring(
            "Xác nhận cuối cùng",
            "Vui lòng nhập 'DELETE ALL' để xác nhận xóa toàn bộ dữ liệu:"
        )
        if confirm2 != "DELETE ALL":
            messagebox.showinfo("Đã hủy", "Hành động xóa đã được hủy")
            return
        
        try:
            data_dir = settings.paths.data_dir
            if data_dir.exists():
                shutil.rmtree(data_dir)
                data_dir.mkdir(parents=True, exist_ok=True)
            
            for f in [settings.paths.embedding_file, settings.paths.names_file]:
                if f.exists():
                    f.unlink()
            
            self.face_detector.known_embeddings = []
            self.face_detector.known_names = []
            self.populate_person_list()
            self.show_video_view()
            
            messagebox.showinfo("Thành công", "Đã xóa toàn bộ dữ liệu")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")

def run_gui(camera, face_detector, state_manager):
    """Run the GUI"""
    root = ctk.CTk()
    app = FaceManagerGUI(root, camera, face_detector, state_manager)
    root.mainloop()