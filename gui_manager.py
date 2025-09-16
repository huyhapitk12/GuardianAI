import customtkinter as ctk
from tkinter import filedialog, simpledialog, messagebox
import os
import shutil
import cv2
import pickle
from PIL import Image

# --- Import thật từ dự án của bạn ---
from detection_core import update_known_data, update_model, app
from config import EMBEDDING_FILE, NAMES_FILE, DATA_DIR
# ------------------------------------

# --- Cài đặt giao diện ---
ctk.set_appearance_mode("Dark") # Chế độ: "System", "Dark", "Light"
ctk.set_default_color_theme("blue") # Theme màu

class FaceManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Guardian - Face Manager")
        self.root.geometry("1000x600")
        self.root.minsize(800, 500)

        self.current_person = None
        self.thumbnail_size = (100, 100)
        self.main_image_size = (350, 350)

        # --- Cấu hình layout chính ---
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- Frame bên trái (Sidebar) ---
        self.left_frame = ctk.CTkFrame(self.root, width=250, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsw")
        self.left_frame.grid_rowconfigure(1, weight=1)

        title_label = ctk.CTkLabel(self.left_frame, text="Danh sách", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.person_list_frame = ctk.CTkScrollableFrame(self.left_frame, label_text="Người đã lưu")
        self.person_list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Frame cho các nút chức năng chung
        general_actions_frame = ctk.CTkFrame(self.left_frame)
        general_actions_frame.grid(row=2, column=0, padx=10, pady=10, sticky="sew")

        model_label = ctk.CTkLabel(general_actions_frame, text="Chọn Model")
        model_label.pack(pady=(10,0), padx=10, fill="x")

        self.model_var = ctk.StringVar(value="buffalo_l")  # default
        model_options = ["buffalo_s", "buffalo_l"]
        model_combo = ctk.CTkOptionMenu(
            general_actions_frame,
            variable=self.model_var,
            values=model_options,
            command=self.change_model
        )
        model_combo.pack(pady=5, padx=10, fill="x")

        btn_add_person = ctk.CTkButton(general_actions_frame, text="Thêm Người Mới", command=self.add_person)
        btn_add_person.pack(pady=5, padx=10, fill="x")

        btn_rebuild = ctk.CTkButton(general_actions_frame, text="Xây Dựng Lại Tất Cả", command=self.rebuild_all)
        btn_rebuild.pack(pady=5, padx=10, fill="x")

        btn_delete_all = ctk.CTkButton(general_actions_frame, text="Xóa Tất Cả", fg_color="#D32F2F", hover_color="#B71C1C", command=self.delete_all)
        btn_delete_all.pack(pady=(5,10), padx=10, fill="x")

        # --- Frame bên phải (Content) ---
        self.right_frame = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

        # --- Populate danh sách ban đầu ---
        self.populate_person_list()
        self.show_placeholder_content()

    def change_model(self, selected):
        update_model(selected)
        messagebox.showinfo("Model Changed", f"Đã chuyển sang model: {selected}")


    def populate_person_list(self):
        # Xóa danh sách cũ
        for widget in self.person_list_frame.winfo_children():
            widget.destroy()

        try:
            persons = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
            for person_name in persons:
                btn = ctk.CTkButton(
                    self.person_list_frame,
                    text=person_name,
                    command=lambda name=person_name: self.select_person(name),
                    fg_color="transparent",
                    anchor="w",  # SỬA LỖI: Dùng anchor="w" thay cho text_align="left"
                    hover_color=("gray85", "gray20")
                )
                btn.pack(fill="x", padx=5, pady=2)
        except FileNotFoundError:
            os.makedirs(DATA_DIR, exist_ok=True)

    def select_person(self, name):
        self.current_person = name
        # Xóa nội dung cũ
        for widget in self.right_frame.winfo_children():
            widget.destroy()

        # --- Header của content ---
        header_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=1)

        person_name_label = ctk.CTkLabel(header_frame, text=name, font=ctk.CTkFont(size=28, weight="bold"))
        person_name_label.grid(row=0, column=0, sticky="w")

        # Nút chức năng cho người được chọn
        person_actions_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        person_actions_frame.grid(row=0, column=1, sticky="e")

        btn_add_img = ctk.CTkButton(person_actions_frame, text="Thêm Ảnh", command=lambda: self.add_image_for_person(name))
        btn_add_img.pack(side="left", padx=5)
        btn_del_person = ctk.CTkButton(person_actions_frame, text="Xóa Người Này", fg_color="#D32F2F", hover_color="#B71C1C", command=lambda: self.delete_person(name))
        btn_del_person.pack(side="left", padx=5)

        # --- Khu vực hiển thị ảnh ---
        self.main_image_label = ctk.CTkLabel(self.right_frame, text="")
        self.main_image_label.grid(row=1, column=0, sticky="nsew", pady=10)

        # --- Thanh cuộn ảnh thumbnail ---
        self.thumbnail_frame = ctk.CTkScrollableFrame(self.right_frame, label_text="Ảnh đã lưu", height=120, orientation="horizontal")
        self.thumbnail_frame.grid(row=2, column=0, sticky="ew", pady=10)

        self.load_person_images(name)

    def show_placeholder_content(self):
        # Xóa nội dung cũ
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        
        placeholder_label = ctk.CTkLabel(self.right_frame, text="Chọn một người từ danh sách\nhoặc thêm người mới",
                                         font=ctk.CTkFont(size=18), text_color="gray50")
        placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
        self.current_person = None

    def load_person_images(self, name):
        person_dir = os.path.join(DATA_DIR, name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Xóa thumbnail cũ
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()

        if not image_files:
            self.main_image_label.configure(image=None, text="Không có ảnh nào cho người này.")
            return

        # Hiển thị ảnh đầu tiên làm ảnh chính
        first_image_path = os.path.join(person_dir, image_files[0])
        self.update_main_image(first_image_path)

        # Tạo thumbnails
        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            try:
                img = Image.open(img_path)
                img.thumbnail(self.thumbnail_size)
                ctk_img = ctk.CTkImage(img, size=self.thumbnail_size)
                
                thumb_btn = ctk.CTkButton(
                    self.thumbnail_frame,
                    text="",
                    image=ctk_img,
                    width=self.thumbnail_size[0],
                    height=self.thumbnail_size[1],
                    command=lambda p=img_path: self.update_main_image(p)
                )
                thumb_btn.pack(side="left", padx=5, pady=5)
            except Exception as e:
                print(f"Không thể tạo thumbnail cho {img_file}: {e}")

    def update_main_image(self, image_path):
        try:
            img = Image.open(image_path)
            ctk_img = ctk.CTkImage(img, size=self.main_image_size)
            self.main_image_label.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"Không thể hiển thị ảnh chính {image_path}: {e}")
            self.main_image_label.configure(image=None, text="Lỗi khi tải ảnh.")

    # --- CÁC HÀM CHỨC NĂNG (LOGIC) ---
    # (Các hàm logic từ add_person đến process_and_save_embeddings giữ nguyên như cũ)

    def add_person(self):
        name = simpledialog.askstring("Thêm Người Mới", "Nhập tên người cần thêm:", parent=self.root)
        if not name or not name.strip():
            if name is not None: messagebox.showwarning("Cảnh báo", "Tên không được để trống.")
            return
        
        name = name.strip()
        person_dir = os.path.join(DATA_DIR, name)
        if os.path.exists(person_dir):
            messagebox.showerror("Lỗi", f"Người có tên '{name}' đã tồn tại.")
            return
        
        img_path = filedialog.askopenfilename(
            title=f"Chọn ảnh đầu tiên cho {name}",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not img_path: return

        os.makedirs(person_dir, exist_ok=True)
        shutil.copy(img_path, person_dir)
        
        self.process_and_save_embeddings(rebuild_mode=False)
        self.populate_person_list()
        self.select_person(name)
        messagebox.showinfo("Thành công", f"Đã thêm '{name}' thành công.")

    def add_image_for_person(self, name):
        img_path = filedialog.askopenfilename(
            title=f"Chọn ảnh mới cho {name}",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )
        if not img_path: return

        person_dir = os.path.join(DATA_DIR, name)
        shutil.copy(img_path, person_dir)
        
        self.process_and_save_embeddings(rebuild_mode=False)
        self.load_person_images(name) # Chỉ cần load lại ảnh cho người hiện tại
        messagebox.showinfo("Thành công", f"Đã thêm ảnh mới cho '{name}'.")

    def delete_person(self, name):
        if not messagebox.askyesno("Xác nhận", f"Bạn có chắc chắn muốn xóa tất cả dữ liệu của '{name}'?"):
            return
        
        try:
            shutil.rmtree(os.path.join(DATA_DIR, name))
            self.process_and_save_embeddings(rebuild_mode=True) # Xóa xong phải build lại
            self.populate_person_list()
            self.show_placeholder_content()
            messagebox.showinfo("Thành công", f"Đã xóa '{name}'.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi xóa: {e}")

    def rebuild_all(self):
        if not messagebox.askyesno("Xác nhận", "Quá trình này sẽ mã hóa lại tất cả các khuôn mặt. Điều này có thể mất thời gian. Bạn có muốn tiếp tục?"):
            return
        
        count = self.process_and_save_embeddings(rebuild_mode=True)
        messagebox.showinfo("Hoàn tất", f"Đã xây dựng lại dữ liệu thành công.\nTổng số {count} khuôn mặt đã được mã hóa.")
        if self.current_person:
            self.select_person(self.current_person)
        else:
            self.show_placeholder_content()

    def delete_all(self):
        confirm1 = messagebox.askyesno("CẢNH BÁO", "Bạn có chắc chắn muốn XÓA TẤT CẢ dữ liệu khuôn mặt không? Hành động này KHÔNG THỂ hoàn tác.", icon='warning')
        if not confirm1: return
        
        confirm2 = simpledialog.askstring("Xác nhận cuối cùng", "Vui lòng nhập 'DELETE ALL' để xác nhận xóa toàn bộ dữ liệu:")
        if confirm2 != "DELETE ALL":
            messagebox.showinfo("Đã hủy", "Hành động xóa đã được hủy.")
            return
            
        try:
            if os.path.exists(DATA_DIR): shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR)
            if os.path.exists(EMBEDDING_FILE): os.remove(EMBEDDING_FILE)
            if os.path.exists(NAMES_FILE): os.remove(NAMES_FILE)
            
            update_known_data()
            self.populate_person_list()
            self.show_placeholder_content()
            messagebox.showinfo("Thành công", "Đã xóa toàn bộ dữ liệu.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi xóa toàn bộ dữ liệu: {e}")

    def process_and_save_embeddings(self, rebuild_mode=True):
        print("Bắt đầu quá trình mã hóa...")
        known_embeddings = []
        known_names = []

        for person_name in os.listdir(DATA_DIR):
            person_path = os.path.join(DATA_DIR, person_name)
            if not os.path.isdir(person_path): continue

            for filename in os.listdir(person_path):
                if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): continue

                filepath = os.path.join(person_path, filename)
                img = cv2.imread(filepath)
                if img is None:
                    print(f"Lỗi: không thể đọc ảnh {filepath}")
                    continue

                faces = app.get(img) # type: ignore
                if faces:
                    embedding = faces[0].embedding
                    known_embeddings.append(embedding)
                    known_names.append(person_name)
                    print(f"Đã mã hóa: {person_name}/{filename}")
                else:
                    print(f"Cảnh báo: Không tìm thấy khuôn mặt trong {filepath}")

        with open(EMBEDDING_FILE, 'wb') as f: pickle.dump(known_embeddings, f)
        with open(NAMES_FILE, 'wb') as f: pickle.dump(known_names, f)
        
        update_known_data()
        print(f"Quá trình mã hóa hoàn tất. Đã lưu {len(known_names)} vector.")
        return len(known_names)


if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceManagerApp(root)
    root.mainloop()