import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image

# Fixed 16-color palette (PICO-8 inspired for high contrast & visibility)
PALETTE = [
    "#000000", "#1D2B53", "#7E2553", "#008751",
    "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
    "#FF004D", "#FFA300", "#FFEC27", "#00E436",
    "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"
]

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

PALETTE_RGB = [hex_to_rgb(c) for c in PALETTE]


class SoundModemApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acoustic Data & Image Transceiver")
        self.geometry("900x680")
        self.configure(bg="#1e1e2e")
        self.resizable(False, False)

        # Shared container for page switching
        self.container = tk.Frame(self, bg="#1e1e2e")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (HomePage, SendPage, ReceivePage):
            frame = F(parent=self.container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


# ==========================================
# 1. HOME SCREEN
# ==========================================
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e2e")
        self.controller = controller

        title = tk.Label(
            self, text="SONIC MODEM", font=("Helvetica", 32, "bold"),
            fg="#cdd6f4", bg="#1e1e2e"
        )
        title.pack(pady=(120, 10))

        subtitle = tk.Label(
            self, text="Acoustic Text & 32x32 Pixel Art Transceiver",
            font=("Helvetica", 14), fg="#a6adc8", bg="#1e1e2e"
        )
        subtitle.pack(pady=(0, 60))

        btn_frame = tk.Frame(self, bg="#1e1e2e")
        btn_frame.pack()

        send_btn = tk.Button(
            btn_frame, text="TRANSMIT (SEND)", font=("Helvetica", 16, "bold"),
            width=18, height=3, bg="#89b4fa", fg="#11111b",
            highlightbackground="#1e1e2e", cursor="hand2",
            command=lambda: controller.show_frame("SendPage")
        )
        send_btn.grid(row=0, column=0, padx=20)

        recv_btn = tk.Button(
            btn_frame, text="LISTEN (RECEIVE)", font=("Helvetica", 16, "bold"),
            width=18, height=3, bg="#a6e3a1", fg="#11111b",
            highlightbackground="#1e1e2e", cursor="hand2",
            command=lambda: controller.show_frame("ReceivePage")
        )
        recv_btn.grid(row=0, column=1, padx=20)


# ==========================================
# 2. SENDING PAGE (Text, File, & 32x32 Editor)
# ==========================================
class SendPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e2e")
        self.controller = controller

        # Top Navigation Bar
        nav = tk.Frame(self, bg="#181825", height=50)
        nav.pack(fill="x")
        tk.Button(
            nav, text="← Home", font=("Helvetica", 12, "bold"),
            bg="#313244", fg="#11111b", highlightbackground="#181825",
            command=lambda: controller.show_frame("HomePage")
        ).pack(side="left", padx=15, pady=10)
        tk.Label(
            nav, text="TRANSMIT DATA / IMAGE", font=("Helvetica", 14, "bold"),
            bg="#181825", fg="#cdd6f4"
        ).pack(side="left", padx=15)

        # Notebook (Tabs for Text, File, Image Editor)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 11, "bold"), padding=[15, 8])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_text = tk.Frame(self.notebook, bg="#181825")
        self.tab_file = tk.Frame(self.notebook, bg="#181825")
        self.tab_editor = tk.Frame(self.notebook, bg="#181825")

        self.notebook.add(self.tab_text, text="  1. Text Message  ")
        self.notebook.add(self.tab_file, text="  2. Choose File  ")
        self.notebook.add(self.tab_editor, text="  3. 32x32 Pixel Editor  ")

        self._build_text_tab()
        self._build_file_tab()
        self._build_editor_tab()

        # Bottom Transmit Controls
        bottom = tk.Frame(self, bg="#181825", height=70)
        bottom.pack(fill="x", side="bottom")

        self.status_lbl = tk.Label(
            bottom, text="Ready to transmit.", font=("Helvetica", 11),
            bg="#181825", fg="#a6adc8"
        )
        self.status_lbl.pack(side="left", padx=20, pady=15)

        self.tx_btn = tk.Button(
            bottom, text="🔊 TRANSMIT ACTIVE TAB", font=("Helvetica", 13, "bold"),
            bg="#f38ba8", fg="#11111b", highlightbackground="#181825",
            padx=20, pady=8, cursor="hand2",
            command=self.start_transmission
        )
        self.tx_btn.pack(side="right", padx=20, pady=10)

    # --- TAB 1: TEXT ---
    def _build_text_tab(self):
        tk.Label(
            self.tab_text, text="Enter UTF-8 Text Payload:",
            font=("Helvetica", 12, "bold"), bg="#181825", fg="#cdd6f4"
        ).pack(anchor="w", padx=20, pady=(20, 5))

        self.text_input = tk.Text(
            self.tab_text, height=14, font=("Courier", 13),
            bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4", relief="flat"
        )
        self.text_input.pack(fill="x", padx=20, pady=10)
        self.text_input.insert("1.0", "Hello World! Testing acoustic FSK modem.")

    # --- TAB 2: FILE PICKER ---
    def _build_file_tab(self):
        self.selected_file_path = None
        self.raw_file_bytes = None

        frame = tk.Frame(self.tab_file, bg="#181825")
        frame.pack(expand=True)

        tk.Button(
            frame, text="📁 Browse Device for File...", font=("Helvetica", 14, "bold"),
            bg="#89b4fa", fg="#11111b", highlightbackground="#181825",
            padx=20, pady=10, cursor="hand2",
            command=self.choose_file
        ).pack(pady=10)

        self.file_info_lbl = tk.Label(
            frame, text="No file selected.", font=("Helvetica", 12),
            bg="#181825", fg="#a6adc8"
        )
        self.file_info_lbl.pack(pady=10)

        self.convert_img_btn = tk.Button(
            frame, text="🎨 Load Image into 32x32 Pixel Editor",
            font=("Helvetica", 11, "bold"), bg="#cba6f7", fg="#11111b",
            highlightbackground="#181825", padx=15, pady=6, state="disabled",
            command=self.load_file_into_editor
        )
        self.convert_img_btn.pack(pady=10)

    def choose_file(self):
        path = filedialog.askopenfilename(title="Select a file to transmit")
        if not path:
            return
        self.selected_file_path = path
        with open(path, "rb") as f:
            self.raw_file_bytes = f.read()

        size_kb = len(self.raw_file_bytes) / 1024
        self.file_info_lbl.config(
            text=f"Selected: {path.split('/')[-1]} ({len(self.raw_file_bytes)} bytes / {size_kb:.2f} KB)"
        )

        # Enable image conversion if it's an image extension
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            self.convert_img_btn.config(state="normal")
        else:
            self.convert_img_btn.config(state="disabled")

    def load_file_into_editor(self):
        """Downscales selected image to 32x32 and snaps each pixel to nearest 16-color palette."""
        try:
            img = Image.open(self.selected_file_path).convert("RGB")
            img = img.resize((32, 32), Image.Resampling.NEAREST)
            for r in range(32):
                for c in range(32):
                    rgb = img.getpixel((c, r))
                    # Find closest color in our 16-color palette using Euclidean distance
                    closest_idx = min(
                        range(16),
                        key=lambda i: sum((rgb[k] - PALETTE_RGB[i][k]) ** 2 for k in range(3))
                    )
                    self.set_grid_pixel(r, c, closest_idx)
            self.notebook.select(self.tab_editor)
            self.status_lbl.config(text="Converted image to 32x32 16-color canvas!")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not process image:\n{e}")

    # --- TAB 3: 32x32 PIXEL EDITOR ---
    def _build_editor_tab(self):
        self.grid_size = 32
        self.cell_size = 13  # 32 * 13 = 416x416 pixel canvas
        self.current_color_idx = 8  # Default red (#FF004D)
        # 32x32 matrix storing palette indices (0-15)
        self.pixel_data = [[0 for _ in range(32)] for _ in range(32)]
        self.rect_ids = [[None for _ in range(32)] for _ in range(32)]

        editor_container = tk.Frame(self.tab_editor, bg="#181825")
        editor_container.pack(expand=True, fill="both", padx=20, pady=10)

        # Left side: 32x32 Canvas
        canvas_size = self.grid_size * self.cell_size
        self.canvas = tk.Canvas(
            editor_container, width=canvas_size, height=canvas_size,
            bg="#000000", highlightthickness=2, highlightbackground="#585b70"
        )
        self.canvas.pack(side="left", padx=(20, 30), pady=5)

        # Initialize grid rectangles
        for r in range(32):
            for c in range(32):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=PALETTE[0], outline="#1e1e2e", width=1
                )
                self.rect_ids[r][c] = rect

        # Bind mouse click and drag to paint
        self.canvas.bind("<Button-1>", self._paint_event)
        self.canvas.bind("<B1-Motion>", self._paint_event)
        # Right click (or Control-Click on Mac trackpad) to erase (set to color 0)
        self.canvas.bind("<Button-2>", self._erase_event)
        self.canvas.bind("<B2-Motion>", self._erase_event)
        self.canvas.bind("<Button-3>", self._erase_event)
        self.canvas.bind("<B3-Motion>", self._erase_event)

        # Right side: 16-Color Palette & Tools
        tools_frame = tk.Frame(editor_container, bg="#181825")
        tools_frame.pack(side="left", fill="y", pady=10)

        tk.Label(
            tools_frame, text="16-Color Palette (4-bit)",
            font=("Helvetica", 12, "bold"), bg="#181825", fg="#cdd6f4"
        ).pack(anchor="w", pady=(0, 10))

        palette_grid = tk.Frame(tools_frame, bg="#181825")
        palette_grid.pack()

        # Use tk.Canvas swatches so macOS renders the exact hex background colors
        self.swatch_widgets = []
        for idx, hex_color in enumerate(PALETTE):
            is_selected = (idx == self.current_color_idx)
            swatch = tk.Canvas(
                palette_grid, width=38, height=32, bg=hex_color,
                highlightthickness=3,
                highlightbackground="#ffffff" if is_selected else "#181825",
                cursor="hand2"
            )
            swatch.grid(row=idx // 4, column=idx % 4, padx=4, pady=4)
            swatch.bind("<Button-1>", lambda e, i=idx: self.select_color(i))
            self.swatch_widgets.append(swatch)

        self.active_color_lbl = tk.Label(
            tools_frame,
            text=f"Selected: Color #{self.current_color_idx} ({PALETTE[self.current_color_idx]})",
            font=("Helvetica", 10), bg="#181825", fg="#a6adc8"
        )
        self.active_color_lbl.pack(pady=10)

        # Quick Utility Buttons
        tk.Button(
            tools_frame, text="Fill Canvas with Selected", font=("Helvetica", 10, "bold"),
            highlightbackground="#181825", width=22, pady=5, cursor="hand2",
            command=self.fill_canvas
        ).pack(pady=4)

        tk.Button(
            tools_frame, text="Clear Canvas (Black)", font=("Helvetica", 10, "bold"),
            highlightbackground="#181825", width=22, pady=5, cursor="hand2",
            command=lambda: self.fill_canvas(0)
        ).pack(pady=4)

        tk.Button(
            tools_frame, text="Load Test Pattern", font=("Helvetica", 10, "bold"),
            highlightbackground="#181825", width=22, pady=5, cursor="hand2",
            command=self.load_test_pattern
        ).pack(pady=4)

    def select_color(self, idx):
        self.current_color_idx = idx
        for i, swatch in enumerate(self.swatch_widgets):
            swatch.config(highlightbackground="#ffffff" if i == idx else "#181825")
        self.active_color_lbl.config(text=f"Selected: Color #{idx} ({PALETTE[idx]})")

    def set_grid_pixel(self, r, c, color_idx):
        if 0 <= r < 32 and 0 <= c < 32:
            self.pixel_data[r][c] = color_idx
            self.canvas.itemconfig(self.rect_ids[r][c], fill=PALETTE[color_idx])

    def _paint_event(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        self.set_grid_pixel(r, c, self.current_color_idx)

    def _erase_event(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        self.set_grid_pixel(r, c, 0)

    def fill_canvas(self, color_idx=None):
        idx = self.current_color_idx if color_idx is None else color_idx
        for r in range(32):
            for c in range(32):
                self.set_grid_pixel(r, c, idx)

    def load_test_pattern(self):
        """Generates a crisp 16-color test pattern for quick modem debugging."""
        for r in range(32):
            for c in range(32):
                color_idx = ((r // 8) * 4 + (c // 8)) % 16
                self.set_grid_pixel(r, c, color_idx)

    def get_packed_canvas_bytes(self) -> bytes:
        """
        Packs the 32x32 4-bit color indices (0..15) into 512 bytes (2 pixels per byte).
        Ready to pass directly to protocol.py!
        """
        flat_nibbles = [self.pixel_data[r][c] for r in range(32) for c in range(32)]
        packed = bytearray()
        for i in range(0, len(flat_nibbles), 2):
            high_nibble = flat_nibbles[i] & 0x0F
            low_nibble = flat_nibbles[i + 1] & 0x0F
            packed.append((high_nibble << 4) | low_nibble)
        return bytes(packed)

    # --- TRANSMIT HOOK ---
    def start_transmission(self):
        active_tab = self.notebook.index(self.notebook.select())

        if active_tab == 0:  # Text
            text = self.text_input.get("1.0", "end-1c").strip()
            payload = text.encode("utf-8")
            self.status_lbl.config(text=f"Transmitting TEXT ({len(payload)} bytes)...", fg="#a6e3a1")
            # TODO: Call protocol.transmit(payload_type="TEXT", data=payload)

        elif active_tab == 1:  # File
            if not self.raw_file_bytes:
                messagebox.showwarning("No File", "Please choose a file first.")
                return
            self.status_lbl.config(text=f"Transmitting FILE ({len(self.raw_file_bytes)} bytes)...", fg="#a6e3a1")
            # TODO: Call protocol.transmit(payload_type="FILE", data=self.raw_file_bytes)

        elif active_tab == 2:  # 32x32 Image
            packed_img = self.get_packed_canvas_bytes()  # Exactly 512 bytes!
            self.status_lbl.config(text=f"Transmitting 32x32 IMAGE ({len(packed_img)} bytes)...", fg="#a6e3a1")
            # TODO: Call protocol.transmit(payload_type="IMAGE", data=packed_img)


# ==========================================
# 3. RECEIVING PAGE
# ==========================================
class ReceivePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e2e")
        self.controller = controller
        self.is_listening = False

        # Top Nav
        nav = tk.Frame(self, bg="#181825", height=50)
        nav.pack(fill="x")
        tk.Button(
            nav, text="← Home", font=("Helvetica", 12, "bold"),
            bg="#313244", fg="#11111b", highlightbackground="#181825",
            command=lambda: controller.show_frame("HomePage")
        ).pack(side="left", padx=15, pady=10)
        tk.Label(
            nav, text="RECEIVER & LIVE DECODER", font=("Helvetica", 14, "bold"),
            bg="#181825", fg="#cdd6f4"
        ).pack(side="left", padx=15)

        main_body = tk.Frame(self, bg="#1e1e2e")
        main_body.pack(fill="both", expand=True, padx=25, pady=20)

        # Left Column: Progressive 32x32 Image Renderer
        left_col = tk.Frame(main_body, bg="#181825", padx=15, pady=15)
        left_col.pack(side="left", fill="y", padx=(0, 15))

        tk.Label(
            left_col, text="Live 32x32 Image Feed", font=("Helvetica", 12, "bold"),
            bg="#181825", fg="#cdd6f4"
        ).pack(pady=(0, 10))

        self.rx_cell_size = 12
        canvas_dim = 32 * self.rx_cell_size
        self.rx_canvas = tk.Canvas(
            left_col, width=canvas_dim, height=canvas_dim,
            bg="#000000", highlightthickness=2, highlightbackground="#585b70"
        )
        self.rx_canvas.pack()

        self.rx_rects = [[None for _ in range(32)] for _ in range(32)]
        for r in range(32):
            for c in range(32):
                x1, y1 = c * self.rx_cell_size, r * self.rx_cell_size
                x2, y2 = x1 + self.rx_cell_size, y1 + self.rx_cell_size
                self.rx_rects[r][c] = self.rx_canvas.create_rectangle(
                    x1, y1, x2, y2, fill="#000000", outline="#11111b"
                )

        # Right Column: Controls, Detected Frequency, & Text Output
        right_col = tk.Frame(main_body, bg="#181825", padx=15, pady=15)
        right_col.pack(side="left", fill="both", expand=True)

        self.listen_btn = tk.Button(
            right_col, text="🎙️ START LISTENING", font=("Helvetica", 14, "bold"),
            bg="#a6e3a1", fg="#11111b", highlightbackground="#181825", pady=10,
            cursor="hand2", command=self.toggle_listening
        )
        self.listen_btn.pack(fill="x", pady=(0, 15))

        self.freq_lbl = tk.Label(
            right_col, text="Detected Peak Freq: --- Hz",
            font=("Courier", 13, "bold"), bg="#313244", fg="#89dceb", pady=8
        )
        self.freq_lbl.pack(fill="x", pady=(0, 15))

        tk.Label(
            right_col, text="Decoded Text / Packet Log:",
            font=("Helvetica", 11, "bold"), bg="#181825", fg="#cdd6f4"
        ).pack(anchor="w")

        self.rx_text_box = tk.Text(
            right_col, height=12, font=("Courier", 11),
            bg="#313244", fg="#cdd6f4", relief="flat"
        )
        self.rx_text_box.pack(fill="both", expand=True, pady=(5, 0))

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.listen_btn.config(text="⏹️ STOP LISTENING")
            self.rx_text_box.insert("end", "[INFO] Microphone stream started...\n")
            # TODO: Start audio_modem receiver thread here
        else:
            self.listen_btn.config(text="🎙️ START LISTENING")
            self.rx_text_box.insert("end", "[INFO] Stopped listening.\n")
            # TODO: Stop audio_modem receiver thread here

    def update_rx_pixel(self, row, col, color_idx):
        """Call this from the decoder thread as nibbles arrive to paint live!"""
        if 0 <= row < 32 and 0 <= col < 32 and 0 <= color_idx < 16:
            self.rx_canvas.itemconfig(self.rx_rects[row][col], fill=PALETTE[color_idx])


if __name__ == "__main__":
    app = SoundModemApp()
    app.mainloop()