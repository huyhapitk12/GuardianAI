"""Persons management panel"""

from __future__ import annotations
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
from gui.widgets import log_activity


class PersonsPanel(ctk.CTkFrame):
    """Persons/faces management panel"""
    
    __slots__ = (
        'face_detector', 'person_list', 'details_panel',
        'search_entry', 'current_person'
    )
    
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
        search_frame, self.search_entry = create_entry(list_panel, "Search...", "🔍")
        search_frame.pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        self.search_entry.bind("<KeyRelease>", lambda _: self.refresh_list())
        
        ctk.CTkLabel(
            list_panel, text="Registered Persons",
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
            list_panel, "➕ Add Person", "primary",
            command=self.add_person
        ).pack(fill="x", padx=Sizes.MD, pady=Sizes.MD)
        
        create_button(
            list_panel, "🔄 Rebuild Embeddings", "secondary",
            command=self.rebuild_embeddings
        ).pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.MD))
        
        create_button(
            list_panel, "🗑️ Delete All Faces", "danger",
            command=self.delete_all_faces
        ).pack(fill="x", padx=Sizes.MD, pady=(0, Sizes.MD))
        
        # Details panel
        self.details_panel = create_card(self)
        self.details_panel.grid(row=0, column=1, sticky="nsew", padx=Sizes.SM, pady=Sizes.SM)
        
        ctk.CTkLabel(
            self.details_panel, text="👤 Select a person",
            font=Fonts.BODY, text_color=Colors.TEXT_MUTED
        ).pack(expand=True)
    
    def refresh_list(self, search: str = None):
        """Refresh person list"""
        # Clear list
        for widget in self.person_list.winfo_children():
            widget.destroy()
        
        faces_dir = settings.paths.faces_dir
        if not faces_dir.exists():
            ctk.CTkLabel(
                self.person_list, text="No persons registered",
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
                self.person_list, text="No matches found",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(pady=Sizes.LG)
            return
        
        # Create cards
        for name in persons:
            self.create_person_card(name)
    
    def create_person_card(self, name: str):
        """Create person card"""
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
            card, text=f"{count} images",
            font=Fonts.CAPTION, text_color=Colors.TEXT_MUTED
        ).pack(side="right", padx=Sizes.MD)
    
    def select_person(self, name: str):
        """Select and show person details"""
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
            actions, "➕ Add Photo", "primary", "small",
            command=lambda: self.add_photos(name)
        ).pack(side="left", padx=2)
        
        create_button(
            actions, "🗑️ Delete", "danger", "small",
            command=lambda: self.delete_person(name)
        ).pack(side="left", padx=2)
        
        # Gallery
        self.load_gallery(name)
        
        log_activity(f"Viewing person: {name}", "info")
    
    def load_gallery(self, name: str):
        """Load person image gallery"""
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
    
    def add_person(self):
        """Add new person dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Add Person")
        dialog.geometry("400x200")
        dialog.configure(fg_color=Colors.BG_PRIMARY)
        dialog.transient(self)
        dialog.grab_set()
        
        content = create_card(dialog)
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            content, text="Add New Person",
            font=Fonts.HEADING, text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(Sizes.LG, Sizes.SM))
        
        entry_frame, name_entry = create_entry(content, "Enter name...")
        entry_frame.pack(pady=Sizes.SM, padx=Sizes.LG, fill="x")
        
        def on_add():
            name = name_entry.get().strip()
            if not name:
                CTkMessagebox(title="Error", message="Please enter a name", icon="warning")
                return
            
            person_dir = settings.paths.faces_dir / name
            if person_dir.exists():
                CTkMessagebox(title="Error", message=f"'{name}' already exists", icon="cancel")
                return
            
            dialog.destroy()
            self.after(100, lambda: self.select_photos_for_person(name))
        
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(pady=Sizes.LG)
        
        create_button(btn_frame, "Add", "primary", command=on_add).pack(side="left", padx=Sizes.XS)
        create_button(btn_frame, "Cancel", "secondary", command=dialog.destroy).pack(side="left", padx=Sizes.XS)
        
        name_entry.focus()
    
    def select_photos_for_person(self, name: str):
        """Select photos for new person"""
        paths = filedialog.askopenfilenames(
            title=f"Select photos for {name}",
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
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
                title="Success",
                message=f"Added '{name}' with {saved} images",
                icon="check"
            )
            log_activity(f"Added person: {name}", "success")
    
    def add_photos(self, name: str):
        """Add photos to existing person"""
        paths = filedialog.askopenfilenames(
            title=f"Add photos for {name}",
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
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
        
        log_activity(f"Added photos for: {name}", "info")
    
    def delete_person(self, name: str):
        """Delete person"""
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
                self.details_panel, text="👤 Select a person",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            
            CTkMessagebox(title="Success", message=f"Deleted '{name}'", icon="check")
            log_activity(f"Deleted person: {name}", "warning")
            
        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")
    
    def rebuild_embeddings(self):
        """Rebuild face embeddings"""
        try:
            count = self.face_detector.rebuild_embeddings()
            CTkMessagebox(
                title="Success",
                message=f"Rebuilt {count} embeddings",
                icon="check"
            )
            self.refresh_list()
            log_activity("Rebuilt embeddings", "success")
        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")
    
    def delete_all_faces(self):
        """Delete all face data (DANGEROUS)"""
        result = CTkMessagebox(
            title="⚠️ WARNING: Delete All Faces",
            message="This will DELETE ALL registered persons and face data!\n\nThis action CANNOT be undone.\n\nAre you absolutely sure?",
            icon="warning",
            option_1="Cancel",
            option_2="Yes, Delete All"
        ).get()
        
        if result != "Yes, Delete All":
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
                self.details_panel, text="👤 Select a person",
                font=Fonts.BODY, text_color=Colors.TEXT_MUTED
            ).pack(expand=True)
            
            CTkMessagebox(title="Success", message="All face data deleted", icon="check")
            log_activity("Deleted all face data", "warning")
            
        except Exception as e:
            CTkMessagebox(title="Error", message=str(e), icon="cancel")