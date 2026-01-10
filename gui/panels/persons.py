"""Panel quản lý người dùng"""

import shutil
import cv2
import threading
from pathlib import Path
from tkinter import filedialog

from PIL import Image
import customtkinter as ctk
from customtkinter import CTkImage
from CTkMessagebox import CTkMessagebox

from config import settings
from utils import security
from gui.styles import Colors, Fonts, Sizes, create_button, create_card, create_entry


# Panel quản lý người và khuôn mặt
class PersonsPanel(ctk.CTkFrame):
    
    def __init__(self, parent, face_detector, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        
        self.face_detector = face_detector
        self.current_person = None
        
        self.build_ui()
        self.refresh_list()
    
    def build_ui(self):
        self.grid_columnconfigure(0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # List panel
        list_panel = create_card(self)
        list_panel.grid(row=0, column=0, sticky="nsew", padx=Sizes.SM, pady=Sizes.SM)
        
        # Search
        search_frame, self.search_entry = create_entry(list_panel, "Tìm kiếm...")
        search_frame.pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        self.search_entry.bind("<KeyRelease>", lambda _: self.refresh_list())
        
        ctk.CTkLabel(
            list_panel, text="Người đã đăng ký",
            font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", padx=Sizes.MD, pady=(0, Sizes.SM))
        
        # Person list
        self.person_list = ctk.CTkScrollableFrame(
            list_panel, fg_color=Colors.BG_PRIMARY,
            corner_radius=Sizes.RADIUS_MD
        )
        self.person_list.pack(fill="both", expand=True, padx=Sizes.SM, pady=(0, Sizes.SM))
        
        # Actions
        create_button(
            list_panel, "Thêm người", "primary",
            command=self.add_person
        ).pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        
        create_button(
            list_panel, "Tạo lại Embeddings", "secondary",
            command=self.rebuild_embeddings
        ).pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.MD))
        
        create_button(
            list_panel, "Xóa tất cả khuôn mặt", "danger",
            command=self.delete_all_faces
        ).pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.MD))
        
        # Details panel
        self.details_panel = create_card(self)
        self.details_panel.grid(row=0, column=1, sticky="nsew", padx=Sizes.SM, pady=Sizes.SM)
        
        ctk.CTkLabel(
            self.details_panel, text="Chọn một người",
            font=Fonts.BODY, text_color=Colors.TEXT_MUTED
        ).pack(expand=True)
    
    # Làm mới danh sách người
    def refresh_list(self, search=None):
        # Clear list
        for widget in self.person_list.winfo_children():
            widget.destroy()
        
        faces_dir = settings.paths.faces_dir
        if not faces_dir.exists():
            ctk.CTkLabel(
                self.person_list, text="Chưa có người nào được đăng ký",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.LG)
            return
        
        # Get persons
        search_term = search or self.search_entry.get()
        persons = sorted([d.name for d in faces_dir.iterdir() if d.is_dir()])
        
        if search_term:
            persons = [p for p in persons if search_term.lower() in p.lower()]
        
        if not persons:
            ctk.CTkLabel(
                self.person_list, text="Không tìm thấy kết quả",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.LG)
            return
        
        # Create cards
        for name in persons:
            self.create_person_card(name)
    
    # Tạo thẻ hiển thị người
    def create_person_card(self, name):
        card = ctk.CTkFrame(
            self.person_list,
            fg_color=Colors.BG_TERTIARY,
            corner_radius=Sizes.RADIUS_MD,
            height=60,
            cursor="hand2"
        )
        card.pack(fill="x", pady=Sizes.XS)
        card.pack_propagate(False)
        
        # Hover effects
        card.bind("<Enter>", lambda _: card.configure(fg_color=Colors.BG_ELEVATED))
        card.bind("<Leave>", lambda _: card.configure(fg_color=Colors.BG_TERTIARY))
        card.bind("<Button-1>", lambda _: self.select_person(name))
        
        # Avatar
        avatar = ctk.CTkFrame(
            card, width=40, height=40,
            fg_color=Colors.PRIMARY, corner_radius=20
        )
        avatar.pack(side="left", padx=Sizes.SM, pady=Sizes.SM)
        avatar.pack_propagate(False)
        
        ctk.CTkLabel(
            avatar, text=name[0].upper(),
            font=Fonts.BODY_BOLD, text_color=Colors.TEXT_PRIMARY
        ).pack(expand=True)
        
        # Name
        ctk.CTkLabel(
            card, text=name,
            font=Fonts.BODY, text_color=Colors.TEXT_PRIMARY
        ).pack(side="left", padx=Sizes.SM)
        
        # Image count
        person_dir = settings.paths.faces_dir / name
        count = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))
        
        ctk.CTkLabel(
            card, text=f"{count} ảnh",
            font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED
        ).pack(side="right", padx=Sizes.MD)
    
    # Chọn và hiển thị chi tiết người
    def select_person(self, name):
        self.current_person = name
        
        # Clear details
        for widget in self.details_panel.winfo_children():
            widget.destroy()
        
        # Header
        header = ctk.CTkFrame(self.details_panel, fg_color="transparent")
        header.pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        
        ctk.CTkLabel(
            header, text=name,
            font=Fonts.TITLE_MD, text_color=Colors.TEXT_PRIMARY
        ).pack(side="left")
        
        # Actions
        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="right")
        
        create_button(
            actions, "Thêm ảnh", "primary", "small",
            command=lambda: self.add_photos(name)
        ).pack(side="left", padx=2)
        
        create_button(
            actions, "Xóa", "danger", "small",
            command=lambda: self.delete_person(name)
        ).pack(side="left", padx=2)
        
        # Gallery
        self.load_gallery(name)
        
        print(f"[INFO] Xem người: {name}")
    
    # Tải thư viện ảnh của người
    def load_gallery(self, name):
        gallery = ctk.CTkScrollableFrame(
            self.details_panel,
            fg_color=Colors.BG_PRIMARY,
            corner_radius=Sizes.RADIUS_MD
        )
        gallery.pack(fill="both", expand=True, padx=Sizes.MD, pady=(0, Sizes.MD))
        
        person_dir = settings.paths.faces_dir / name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        
        if not images:
            ctk.CTkLabel(
                gallery, text="No images",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.LG)
            return
        
        # Load images in thread
        def load_thread():
            for i, img_path in enumerate(images):
                if not gallery.winfo_exists():
                    return
                
                try:
                    img = security.load_image(img_path)
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    pil_img.thumbnail((150, 150))
                    
                    def add_to_ui(pil=pil_img, row=i//3, col=i%3):
                        if not gallery.winfo_exists():
                            return
                        
                        ctk_img = CTkImage(pil, size=(150, 150))
                        
                        frame = ctk.CTkFrame(gallery, fg_color=Colors.BG_TERTIARY, corner_radius=Sizes.RADIUS_SM)
                        frame.grid(row=row, column=col, padx=Sizes.SM, pady=Sizes.SM)
                        
                        label = ctk.CTkLabel(frame, text="", image=ctk_img)
                        label.pack(padx=2, pady=2)
                        label.image = ctk_img
                    
                    self.after(0, add_to_ui)
                    
                except Exception:
                    pass
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    # Dialog thêm người mới
    def add_person(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Thêm người")
        dialog.geometry("400x200")
        dialog.configure(fg_color=Colors.BG_PRIMARY)
        dialog.transient(self)
        dialog.grab_set()
        
        content = create_card(dialog)
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            content, text="Thêm người mới",
            font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(Sizes.LG, Sizes.SM))
        
        entry_frame, name_entry = create_entry(content, "Nhập tên...")
        entry_frame.pack(pady=Sizes.SM, padx=Sizes.LG, fill="x")
        
        def on_add():
            name = name_entry.get().strip()
            if not name:
                CTkMessagebox(title="Lỗi", message="Vui lòng nhập tên", icon="warning")
                return
            
            person_dir = settings.paths.faces_dir / name
            if person_dir.exists():
                CTkMessagebox(title="Lỗi", message=f"'{name}' đã tồn tại", icon="cancel")
                return
            
            dialog.destroy()
            self.after(100, lambda: self.select_photos_for_person(name))
        
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(pady=Sizes.LG)
        
        create_button(btn_frame, "Thêm", "primary", command=on_add).pack(side="left", padx=Sizes.XS)
        create_button(btn_frame, "Hủy", "secondary", command=dialog.destroy).pack(side="left", padx=Sizes.XS)
        
        name_entry.focus()
    
    # Chọn ảnh cho người mới
    def select_photos_for_person(self, name):
        paths = filedialog.askopenfilenames(
            title=f"Chọn ảnh cho {name}",
            filetypes=[("Ảnh", "*.jpg *.png *.jpeg")]
        )
        
        if not paths:
            return
        
        person_dir = settings.paths.faces_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        saved = 0
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                dest = person_dir / Path(path).name
                security.save_image(dest, img)
                saved += 1
        
        if saved > 0:
            self.face_detector.rebuild_embeddings()
            self.refresh_list()
            self.select_person(name)
            
            CTkMessagebox(
                title="Thành công",
                message=f"Đã thêm '{name}' với {saved} ảnh",
                icon="check"
            )
            print(f"[OK] Đã thêm người: {name}")
    
    # Thêm ảnh cho người đã có
    def add_photos(self, name):
        paths = filedialog.askopenfilenames(
            title=f"Thêm ảnh cho {name}",
            filetypes=[("Ảnh", "*.jpg *.png *.jpeg")]
        )
        
        if not paths:
            return
        
        person_dir = settings.paths.faces_dir / name
        
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                dest = person_dir / Path(path).name
                security.save_image(dest, img)
        
        self.face_detector.rebuild_embeddings()
        self.load_gallery(name)
        
        print(f"[INFO] Đã thêm ảnh cho: {name}")
    
    # Xóa người
    def delete_person(self, name):
        result = CTkMessagebox(
            title="Confirm Delete",
            message=f"Delete '{name}' and all photos?",
            icon="question",
            option_1="Cancel",
            option_2="Delete"
        ).get()
        
        if result != "Delete":
            return
        
        try:
            person_dir = settings.paths.faces_dir / name
            shutil.rmtree(person_dir)
            
            self.face_detector.rebuild_embeddings()
            self.refresh_list()
            
            # Clear details
            for widget in self.details_panel.winfo_children():
                widget.destroy()
            
            ctk.CTkLabel(
                self.details_panel, text="Chọn một người",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            
            CTkMessagebox(title="Thành công", message=f"Đã xóa '{name}'", icon="check")
            print(f"[WARN] Đã xóa người: {name}")
            
        except Exception as e:
            CTkMessagebox(title="Lỗi", message=str(e), icon="cancel")
    
    # Xây dựng lại vector đặc trưng (embeddings)
    def rebuild_embeddings(self):
        try:
            count = self.face_detector.rebuild_embeddings()
            CTkMessagebox(
                title="Thành công",
                message=f"Đã tạo lại {count} embeddings",
                icon="check"
            )
            self.refresh_list()
            print("[OK] Đã tạo lại embeddings")
        except Exception as e:
            CTkMessagebox(title="Lỗi", message=str(e), icon="cancel")
    
    # Xóa toàn bộ dữ liệu khuôn mặt (NGUY HIỂM)
    def delete_all_faces(self):
        result = CTkMessagebox(
            title="CẢNH BÁO: Xóa toàn bộ",
            message="Thao tác này sẽ XÓA TẤT CẢ người đã đăng ký và dữ liệu khuôn mặt!\n\nKHÔNG THỂ hoàn tác.\n\nBạn chắc chắn chứ?",
            icon="warning",
            option_1="Hủy",
            option_2="Xóa tất cả"
        ).get()
        
        if result != "Xóa tất cả":
            return
        
        try:
            faces_dir = settings.paths.faces_dir
            data_dir = settings.paths.data_dir
            
            # Delete all person folders
            if faces_dir.exists():
                for person_dir in faces_dir.iterdir():
                    if person_dir.is_dir():
                        shutil.rmtree(person_dir)
            
            # Delete embedding files
            emb_file = data_dir / "known_embeddings.pkl"
            names_file = data_dir / "known_names.pkl"
            if emb_file.exists():
                emb_file.unlink()
            if names_file.exists():
                names_file.unlink()
            
            # Reload detector
            self.face_detector.embeddings = []
            self.face_detector.names = []
            
            # Refresh UI
            self.refresh_list()
            for widget in self.details_panel.winfo_children():
                widget.destroy()
            ctk.CTkLabel(
                self.details_panel, text="Chọn một người",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            
            CTkMessagebox(title="Thành công", message="Đã xóa toàn bộ dữ liệu khuôn mặt", icon="check")
            print("[WARN] Đã xóa toàn bộ dữ liệu khuôn mặt")
            
        except Exception as e:
            CTkMessagebox(title="Lỗi", message=str(e), icon="cancel")