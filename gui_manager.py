# gui_manager.py
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import os, shutil, cv2, pickle
from detection_core import app, update_known_data  # insightface app loaded in detection_core
from config import EMBEDDING_FILE, NAMES_FILE, DATA_DIR

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Guardian - Face Manager")
        frame = tk.Frame(self.root); frame.pack(padx=8,pady=8)
        tk.Button(frame, text="Thêm khuôn mặt", command=self.add_face).grid(row=0,column=0)
        tk.Button(frame, text="Xây dựng lại embeddings", command=self.rebuild).grid(row=0,column=1)
        tk.Button(frame, text="Danh sách khuôn mặt", command=self.list_faces).grid(row=0,column=2)
        tk.Button(frame, text="Xóa khuôn mặt", command=self.delete_face).grid(row=0,column=3)
        tk.Button(frame, text="Đóng", command=self.root.destroy).grid(row=0,column=4)

    def add_face(self):
        img = filedialog.askopenfilename(filetypes=[("Image","*.jpg *.png *.jpeg")])
        if not img: return
        name = simpledialog.askstring("Tên","Nhập tên người:", parent=self.root)
        if not name: return
        target = os.path.join(DATA_DIR, name)
        os.makedirs(target, exist_ok=True)
        shutil.copy(img, os.path.join(target, os.path.basename(img)))

        with open(EMBEDDING_FILE, 'rb') as f:
                known_embeddings = pickle.load(f)
                f.close()
        with open(NAMES_FILE, 'rb') as f:
                known_names = pickle.load(f)
                f.close()

        for filename in os.listdir(target):
            if not (filename.endswith(".jpg") or filename.endswith(".png")):
                continue

            filepath = os.path.join(target, filename)
            img = cv2.imread(filepath)

            if img is None:
                print(f"Lỗi: không thể đọc ảnh {filepath}")
                continue

            faces = app.get(img)

            if faces:
                embedding = faces[0].embedding
                known_embeddings.append(embedding)
                known_names.append(name)
                print(f"Đã mã hóa: {name}/{filename}")
            else:
                print(f"Cảnh báo: Không tìm thấy khuôn mặt nào trong ảnh {filepath}")

        if known_embeddings:
            print(f"Đã lưu {len(known_names)} vector đặc trưng vào bộ nhớ cache.")
            with open(EMBEDDING_FILE, 'wb') as f:
                pickle.dump(known_embeddings, f)
                f.close()
            with open(NAMES_FILE, 'wb') as f:
                pickle.dump(known_names, f)
                f.close()
            # Update the detection_core known data
            update_known_data()

    def rebuild(self):
        # simple approach: remove existing embeddings file and call detection_core re-encode logic:
        # Here we reuse detection_core encoding: ask user to rerun program or you can implement function to rebuild
        messagebox.showinfo("Thông báo","Việc xây dựng lại sẽ chạy mã hóa - điều này có thể mất thời gian.")
        known_embeddings = []
        known_names = []

        for person_name in os.listdir(DATA_DIR):
            person_path = os.path.join(DATA_DIR, person_name)
            if not os.path.isdir(person_path):
                continue

            for filename in os.listdir(person_path):
                if not (filename.endswith(".jpg") or filename.endswith(".png")):
                    continue

                filepath = os.path.join(person_path, filename)
                img = cv2.imread(filepath)

                if img is None:
                    messagebox.showerror("Lỗi", f"Không thể đọc ảnh {filepath}")
                    continue

                faces = app.get(img)

                if faces:
                    embedding = faces[0].embedding
                    known_embeddings.append(embedding)
                    known_names.append(person_name)
                    messagebox.showinfo("Thông báo", f"Đã mã hóa: {person_name}/{filename}")
                else:
                    messagebox.showwarning("Cảnh báo", f"Không tìm thấy khuôn mặt nào trong ảnh {filepath}")

        if known_embeddings:
            messagebox.showinfo("Thông báo", f"Đã mã hóa {len(known_names)} khuôn mặt.")
            with open(EMBEDDING_FILE, 'wb') as f:
                pickle.dump(known_embeddings, f)
                f.close()
            with open(NAMES_FILE, 'wb') as f:
                pickle.dump(known_names, f)
                f.close()

        del known_embeddings[:]
        del known_names[:]

        update_known_data()

    def list_faces(self):
        persons = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
        messagebox.showinfo("Faces", "\n".join(persons) if persons else "No faces")

    def delete_face(self):
        persons = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
        if not persons:
            messagebox.showinfo("Xóa","Không có khuôn mặt nào")
            return
        sel = simpledialog.askstring("Xóa","Nhập tên để xóa:\n" + ", ".join(persons), parent=self.root)
        if not sel: return
        p = os.path.join(DATA_DIR, sel)
        if os.path.exists(p):
            shutil.rmtree(p)
            messagebox.showinfo("Đã xóa", f"{sel} đã bị xóa. Vui lòng chạy lại.")
            with open(EMBEDDING_FILE, 'rb') as f:
                known_embeddings = pickle.load(f)
                f.close()
            with open(NAMES_FILE, 'rb') as f:
                known_names = pickle.load(f)
                f.close()

            known_embeddings = [e for i, e in enumerate(known_embeddings) if known_names[i] != sel]
            known_names = [n for n in known_names if n != sel]

            with open(EMBEDDING_FILE, 'wb') as f:
                pickle.dump(known_embeddings, f)
                f.close()
            with open(NAMES_FILE, 'wb') as f:
                pickle.dump(known_names, f)
                f.close()

            del known_embeddings[:]
            del known_names[:]

            update_known_data()
        else:
            messagebox.showerror("Lỗi","Không tìm thấy")

    def run(self):
        self.root.mainloop()
