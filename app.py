import os
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
from PIL import ImageTk

from checker_core import run_checker


class DuplicateCheckerApp(tk.Tk):
    # Brand colours
    TEAL = "#04979e"
    RED = "#ea4e1d"
    YELLOW = "#f9a823"

    # Neutrals (light theme)
    WHITE = "#ffffff"
    INK = "#111111"
    GREY_050 = "#fafafa"
    GREY_100 = "#f5f5f5"
    GREY_200 = "#eeeeee"
    GREY_300 = "#d9d9d9"
    GREY_600 = "#666666"

    def __init__(self):
        super().__init__()

        self.geometry("980x620")
        self.minsize(860, 520)
        self.title("Easy Read Online – AI Duplicate Image Checker")

        # Font (you said everyone has it installed)
        self.base_font = self._pick_font(("FS Me", "FSMe", "FSMe-Regular"))
        # IMPORTANT: tuple form prevents 'expected integer but got "Me"' on macOS
        self.option_add("*Font", (self.base_font, 13))

        self.pdf_path = None
        self.results = []
        self._preview_imgtk = None  # keep ref or Tk will drop it

        self._apply_style()
        self._build_ui()

    def _pick_font(self, candidates):
        # Use self to ensure Tk is fully initialised (helps with Automator launches)
        available = set(tkfont.families(self))
        for f in candidates:
            if f in available:
                return f
        # Fallback just in case
        return "Arial"

    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")  # gives consistent control on macOS

        # Frames / labels
        style.configure("App.TFrame", background=self.WHITE)
        style.configure("Header.TFrame", background=self.WHITE)

        style.configure(
            "Header.TLabel",
            background=self.WHITE,
            foreground=self.INK,
            font=(self.base_font, 18, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background=self.WHITE,
            foreground=self.GREY_600,
            font=(self.base_font, 12),
        )

        style.configure(
            "App.TLabel",
            background=self.WHITE,
            foreground=self.INK,
            font=(self.base_font, 13),
        )
        style.configure(
            "Status.TLabel",
            background=self.WHITE,
            foreground=self.GREY_600,
            font=(self.base_font, 12),
        )

        # Buttons
        style.configure(
            "TButton",
            padding=9,
            font=(self.base_font, 13),
            background=self.GREY_200,
            foreground=self.INK,
        )
        style.map("TButton", background=[("active", self.GREY_300)])

        # Primary action button (Run)
        style.configure(
            "Primary.TButton",
            padding=10,
            font=(self.base_font, 13, "bold"),
            background=self.TEAL,
            foreground=self.WHITE,
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#037f85")],
            foreground=[("active", self.WHITE)],
        )

        # Treeview (table)
        style.configure(
            "Treeview",
            background=self.WHITE,
            fieldbackground=self.WHITE,
            foreground=self.INK,
            rowheight=30,
            bordercolor=self.GREY_300,
            lightcolor=self.GREY_300,
            darkcolor=self.GREY_300,
        )
        style.map("Treeview", background=[("selected", "#d9f3f2")])  # light teal selection

        style.configure(
            "Treeview.Heading",
            background=self.GREY_100,
            foreground=self.INK,
            font=(self.base_font, 12, "bold"),
        )

    def _build_ui(self):
        # Header area
        header = ttk.Frame(self, padding=12, style="Header.TFrame")
        header.pack(fill="x")

        # Title row
        title_row = ttk.Frame(header, style="Header.TFrame")
        title_row.pack(fill="x")

        ttk.Label(title_row, text="AI Duplicate Image Checker", style="Header.TLabel").pack(side="left")

        # Small colour “pops” (like the logo shapes)
        dots = tk.Frame(title_row, bg=self.WHITE)
        dots.pack(side="left", padx=(12, 0))
        tk.Canvas(dots, width=54, height=16, bg=self.WHITE, highlightthickness=0).pack()

        # We draw small shapes to echo the brand
        c = dots.winfo_children()[0]
        # teal square
        c.create_rectangle(2, 3, 12, 13, fill=self.TEAL, outline=self.TEAL)
        # red circle
        c.create_oval(20, 3, 30, 13, fill=self.RED, outline=self.RED)
        # yellow triangle
        c.create_polygon(38, 13, 48, 13, 43, 3, fill=self.YELLOW, outline=self.YELLOW)

        ttk.Label(
            title_row,
            text="Find repeated images fast for amends",
            style="SubHeader.TLabel",
        ).pack(side="left", padx=(10, 0))

        # Accent line
        tk.Frame(self, height=4, bg=self.TEAL).pack(fill="x")

        # Controls row
        controls = ttk.Frame(self, padding=(12, 10, 12, 10), style="Header.TFrame")
        controls.pack(fill="x")

        self.path_label = ttk.Label(controls, text="No PDF selected", style="App.TLabel")
        self.path_label.pack(side="left", padx=(0, 10))

        ttk.Button(controls, text="Choose PDF", command=self.choose_pdf).pack(side="left")
        ttk.Button(controls, text="Run", command=self.run, style="Primary.TButton").pack(side="left", padx=8)
        ttk.Button(controls, text="Reset", command=self.reset).pack(side="left")

        self.status = ttk.Label(controls, text="", style="Status.TLabel")
        self.status.pack(side="right")

        # Body
        body = ttk.Frame(self, padding=12, style="App.TFrame")
        body.pack(fill="both", expand=True)

        main = ttk.PanedWindow(body, orient="horizontal")
        main.pack(fill="both", expand=True)

        # Left panel: table
        left = ttk.Frame(main, style="App.TFrame")
        main.add(left, weight=3)

        columns = ("count", "pages", "cluster_id")  # cluster last (least important)
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=18)

        self.tree.heading("count", text="Count")
        self.tree.heading("pages", text="Pages")
        self.tree.heading("cluster_id", text="Cluster")

        self.tree.column("count", width=80, anchor="center")
        self.tree.column("pages", width=560, anchor="w")
        self.tree.column("cluster_id", width=90, anchor="center")

        self.tree.pack(fill="both", expand=True)

        # zebra rows (light)
        self.tree.tag_configure("even", background=self.WHITE)
        self.tree.tag_configure("odd", background=self.GREY_050)

        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Right panel: preview
        right = ttk.Frame(main, padding=10, style="App.TFrame")
        main.add(right, weight=2)

        ttk.Label(right, text="Preview", style="App.TLabel", font=(self.base_font, 16, "bold")).pack(anchor="w")

        # Preview border box so it never feels “cut off”
        preview_box = tk.Frame(right, bg=self.GREY_300, padx=1, pady=1)
        preview_box.pack(fill="both", expand=True, padx=10, pady=(10, 10))

        preview_inner = tk.Frame(preview_box, bg=self.WHITE)
        preview_inner.pack(fill="both", expand=True)

        self.preview_label = ttk.Label(preview_inner, style="App.TLabel")
        self.preview_label.pack(fill="both", expand=True, padx=12, pady=12)

    def choose_pdf(self):
        path = filedialog.askopenfilename(
            title="Select a PDF",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not path:
            return

        self.pdf_path = path
        self.path_label.config(text=os.path.basename(path))
        self.status.config(text="Ready")

    def run(self):
        if not self.pdf_path:
            messagebox.showinfo("No PDF", "Choose a PDF first.")
            return

        self.status.config(text="Running…")
        self.update_idletasks()

        def worker():
            try:
                results = run_checker(self.pdf_path)
            except Exception as e:
                self.after(0, lambda: self._on_run_error(e))
                return
            self.after(0, lambda: self._on_run_done(results))

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_done(self, results):
        self.results = results

        # clear table
        for row in self.tree.get_children():
            self.tree.delete(row)

        # fill table
        for i, r in enumerate(self.results):
            pages_str = " ".join(map(str, r["pages"]))
            tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert(
                "",
                "end",
                iid=str(i),
                values=(r["count"], pages_str, r["cluster_id"]),
                tags=(tag,),
            )

        self.status.config(text=f"Found {len(self.results)} duplicate groups")

        # auto preview first
        if self.results:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.show_preview(0)
        else:
            self.preview_label.config(image="")
            self._preview_imgtk = None

    def _on_run_error(self, e):
        self.status.config(text="")
        messagebox.showerror("Error", str(e))

    def on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self.show_preview(idx)

    def show_preview(self, idx: int):
        if idx < 0 or idx >= len(self.results):
            return

        pil_img = self.results[idx]["preview_pil"]

        max_w, max_h = 430, 530
        img = pil_img.copy()
        img.thumbnail((max_w, max_h))

        self._preview_imgtk = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_imgtk, anchor="center")

    def reset(self):
        self.pdf_path = None
        self.results = []
        self.path_label.config(text="No PDF selected")
        self.status.config(text="")
        self.preview_label.config(image="")
        self._preview_imgtk = None

        for row in self.tree.get_children():
            self.tree.delete(row)


if __name__ == "__main__":
    app = DuplicateCheckerApp()
    app.mainloop()
