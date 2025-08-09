# gui_manager.py
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import os, shutil
from detection_core import app  # insightface app loaded in detection_core

DATA_DIR = "Data/Image"

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Guardian - Face Manager")
        frame = tk.Frame(self.root); frame.pack(padx=8,pady=8)
        tk.Button(frame, text="Add face (image)", command=self.add_face).grid(row=0,column=0)
        tk.Button(frame, text="Rebuild embeddings", command=self.rebuild).grid(row=0,column=1)
        tk.Button(frame, text="List faces", command=self.list_faces).grid(row=0,column=2)
        tk.Button(frame, text="Delete face", command=self.delete_face).grid(row=0,column=3)
        tk.Button(frame, text="Close", command=self.root.destroy).grid(row=0,column=4)

    def add_face(self):
        img = filedialog.askopenfilename(filetypes=[("Image","*.jpg *.png *.jpeg")])
        if not img: return
        name = simpledialog.askstring("Name","Enter person name:", parent=self.root)
        if not name: return
        target = os.path.join(DATA_DIR, name)
        os.makedirs(target, exist_ok=True)
        shutil.copy(img, os.path.join(target, os.path.basename(img)))
        messagebox.showinfo("OK","Saved. Click Rebuild embeddings to update.")

    def rebuild(self):
        # simple approach: remove existing embeddings file and call detection_core re-encode logic:
        # Here we reuse detection_core encoding: ask user to rerun program or you can implement function to rebuild
        messagebox.showinfo("Note","Rebuild will run encoding - it may take time. Please run main.py to let detection_core create embeddings.")

    def list_faces(self):
        persons = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
        messagebox.showinfo("Faces", "\n".join(persons) if persons else "No faces")

    def delete_face(self):
        persons = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))]
        if not persons:
            messagebox.showinfo("Delete","No faces")
            return
        sel = simpledialog.askstring("Delete","Enter name to delete:\n" + ", ".join(persons), parent=self.root)
        if not sel: return
        p = os.path.join(DATA_DIR, sel)
        if os.path.exists(p):
            shutil.rmtree(p)
            messagebox.showinfo("Deleted", f"{sel} deleted. Run rebuild.")
        else:
            messagebox.showerror("Error","Not found")

    def run(self):
        self.root.mainloop()
