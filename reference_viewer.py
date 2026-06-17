"""
reference_viewer.py — Visor de señas de referencia LSM
=======================================================
Muestra un grid de todas las señas disponibles en el dataset.
Haz click en cualquier seña para ver el video de referencia en loop.

Corre con:
    python reference_viewer.py
"""

import os
import glob
import threading
import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

DATASET_DIR = "LSM_Dataset"

# Mapeo inglés → español para mostrar nombres correctos en el visor
EN_ES = {
    "goodbye":"adios","thank_you":"gracias","cry":"llorar",
    "good_morning":"buenos_dias","good_afternoon":"buenas_tardes",
    "good_night":"buenas_noches","see_you":"hasta_luego",
    "good_person":"buena_persona","bad_person":"mala_persona",
    "i_want_more":"quiero_mas","i_did_not_like":"no_me_gusto",
    "to_silence":"silencio","sign_language":"lenguaje_de_senas",
    "us_male":"nosotros","us_female":"nosotras",
    "they_male":"ellos","they_female":"ellas",
    "you_plural":"ustedes","someone":"alguien",
    "father":"padre","mother":"madre","brother":"hermano",
    "sister":"hermana","husband":"esposo","wife":"esposa",
    "grandfather":"abuelo","grandmother":"abuela",
    "uncle":"tio","aunt":"tia","nephew":"sobrino","niece":"sobrina",
    "cousin_boy":"primo","cousin_girl":"prima",
    "boyfriend":"novio","girlfriend":"novia",
    "baby":"bebe","daughter":"hija","son":"hijo","child":"nino",
    "man":"hombre","women":"mujeres","adult":"adulto","young":"joven",
    "ugly":"feo","pretty":"bonito","strong":"fuerte","weak":"debil",
    "slim":"delgado","fat":"gordo","deaf":"sordo","blind":"ciego","dumb":"mudo",
    "teacher":"maestro","doctor":"medico","secretary":"secretaria",
    "mechanic":"mecanico","carpenter":"carpintero","shoemaker":"zapatero",
    "stylist":"estilista","waiter":"mesero","firefighter":"bombero",
    "police":"policia","actor":"actor","president":"presidente",
    "love":"amor","laugh":"reir","frighten":"asustar",
    "embrace":"abrazo","slap":"bofetada","throw":"lanzar",
    "look_for":"buscar","pick_up":"recoger","forget":"olvidar",
    "believe":"creer","lie":"mentira","keep":"guardar","close":"cerrar",
    "clean":"limpiar","fix":"arreglar","stop":"parar","do":"hacer",
    "eat":"comer","sleep":"dormir","hear":"escuchar","play":"jugar",
    "reading":"lectura","writing":"escritura","sewing":"costura",
    "head":"cabeza","hair":"cabello","face":"cara","eyes":"ojos",
    "ear":"oreja","ears":"orejas","nose":"nariz","mouth":"boca",
    "cheek":"mejilla","beard":"barba","moustache":"bigote",
    "teeth":"dientes","arm":"brazo","feet":"pies",
    "earring":"arete","necklace":"collar","nail":"una",
    "shirt":"camisa","dress":"vestido","blouse":"blusa","pants":"pantalon",
    "shorts":"pantalon_corto","skirt":"falda","shoes":"zapatos",
    "boot":"bota","glove":"guante","underwear":"ropa_interior","suit":"traje",
    "home":"casa","door":"puerta","window":"ventana","floor":"piso",
    "ceiling":"techo","wall":"pared","room":"habitacion","hall":"pasillo",
    "table":"mesa","bed":"cama","cradle":"cuna","curtain":"cortina",
    "lamp":"lampara","stairs":"escaleras","broom":"escoba","pane":"vidrio",
    "kitchen":"cocina","meal":"comida","breakfast":"desayuno","dinner":"cena",
    "dish":"plato","spoon":"cuchara","fork":"tenedor","knife":"cuchillo",
    "glass":"vaso","napkin":"servilleta","salt":"sal","cafe":"cafe",
    "car":"automovil","bus":"autobus","motorcycle":"motocicleta",
    "bicycle":"bicicleta","van":"camioneta","plane":"avion","ship":"barco",
    "train":"tren","truck":"camion","helicopter":"helicoptero","subway":"metro",
    "hospital":"hospital","restaurant":"restaurante","market":"mercado",
    "supermarket":"supermercado","airport":"aeropuerto","downtown":"centro",
    "museum":"museo","cinema":"cine","circus":"circo","bathroom":"bano",
    "building":"edificio","library":"biblioteca","school":"escuela",
    "lesson":"leccion","exam":"examen","qualification":"calificacion",
    "notebook":"cuaderno","pencil":"lapiz","pencil_sharpener":"sacapuntas",
    "eraser":"borrador","ruler":"regla",
    "day":"dia","week":"semana","hour":"hora","minutes":"minutos",
    "seconds":"segundos","monday":"lunes","tuesday":"martes",
    "wednesday":"miercoles","thursday":"jueves","friday":"viernes",
    "saturday":"sabado","sunday":"domingo",
    "january":"enero","february":"febrero","march":"marzo","april":"abril",
    "may":"mayo","june":"junio","july":"julio","august":"agosto",
    "september":"septiembre","october":"octubre","november":"noviembre",
    "december":"diciembre",
    "colors":"colores","high":"alto","low":"bajo","order":"orden","year":"ano",
    "family":"familia","he":"el","she":"ella","you":"tu","i":"yo",
    "help":"ayuda","please":"por_favor",
    "b._c._norte":"baja_california_norte","b._c._sur":"baja_california_sur",
    "mexico":"mexico","michoacan":"michoacan","nuevo_leon":"nuevo_leon",
    "yucatan":"yucatan","queretaro":"queretaro","san_luis_potosi":"san_luis_potosi",
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def find_all_signs():
    """Devuelve dict {nombre_español: [lista de rutas mp4]} del dataset."""
    signs = {}
    for mp4 in glob.glob(os.path.join(DATASET_DIR, "**", "*.mp4"), recursive=True):
        raw_name = os.path.basename(os.path.dirname(mp4))
        # Traducir al español si existe mapeo, si no dejar como está
        display_name = EN_ES.get(raw_name, raw_name)
        signs.setdefault(display_name, []).append(mp4)
    return dict(sorted(signs.items()))


class ReferenceViewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Visor de Referencia LSM")
        self.geometry("1200x750")
        self.resizable(True, True)

        self.signs = find_all_signs()
        self._play_thread = None
        self._stop_flag = threading.Event()
        self._current_sign = None

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # ── Panel izquierdo: lista de señas ─────────────────────────────
        left = ctk.CTkFrame(self, width=280)
        left.grid(row=0, column=0, sticky="nsew", padx=(10,5), pady=10)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.search_var = ctk.StringVar()
        self.search_var.trace_add("write", self._on_search)
        search = ctk.CTkEntry(left, placeholder_text="Buscar seña...",
                              textvariable=self.search_var, height=36)
        search.grid(row=0, column=0, sticky="ew", padx=8, pady=(8,4))

        self.lbl_count = ctk.CTkLabel(left, text=f"{len(self.signs)} señas disponibles",
                                       font=("Inter", 11), text_color="gray")
        self.lbl_count.grid(row=1, column=0, sticky="w", padx=10, pady=(0,4))

        list_frame = ctk.CTkScrollableFrame(left)
        list_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0,8))
        list_frame.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(2, weight=1)

        self._list_frame = list_frame
        self._sign_buttons = {}
        self._populate_list(list(self.signs.keys()))

        # ── Panel derecho: video + info ──────────────────────────────────
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(5,10), pady=10)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Nombre de la seña
        self.lbl_sign = ctk.CTkLabel(right, text="← Selecciona una seña",
                                      font=("Inter", 22, "bold"))
        self.lbl_sign.grid(row=0, column=0, pady=(16,8))

        # Canvas de video
        self.canvas = tk.Canvas(right, bg="black", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=16, pady=8)

        # Info barra inferior
        bottom = ctk.CTkFrame(right, height=48)
        bottom.grid(row=2, column=0, sticky="ew", padx=16, pady=(0,12))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)

        self.lbl_video_info = ctk.CTkLabel(bottom, text="", font=("Inter", 12),
                                            text_color="gray")
        self.lbl_video_info.grid(row=0, column=0, sticky="w", padx=12)

        self.lbl_nav = ctk.CTkLabel(bottom, text="", font=("Inter", 12))
        self.lbl_nav.grid(row=0, column=1, sticky="e", padx=12)

        # Controles de navegación entre videos
        nav_frame = ctk.CTkFrame(right, height=44, fg_color="transparent")
        nav_frame.grid(row=3, column=0, pady=(0,10))

        self.btn_prev = ctk.CTkButton(nav_frame, text="◀ Anterior video",
                                       width=160, command=self._prev_video)
        self.btn_prev.pack(side="left", padx=8)

        self.btn_next = ctk.CTkButton(nav_frame, text="Siguiente video ▶",
                                       width=160, command=self._next_video)
        self.btn_next.pack(side="left", padx=8)

        self._video_idx = 0
        self._tk_image = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Poblar lista después de que la ventana esté completamente renderizada
        self.after(100, lambda: self._populate_list(list(self.signs.keys())))

    def _populate_list(self, sign_names):
        for w in self._list_frame.winfo_children():
            w.destroy()
        self._sign_buttons = {}
        for name in sign_names:
            n_videos = len(self.signs.get(name, []))
            label = f"{name}  ({n_videos}v)"
            btn = ctk.CTkButton(
                self._list_frame, text=label, anchor="w",
                height=32, font=("Inter", 13),
                fg_color="transparent", hover_color="#2A2D3E",
                text_color="white",
                command=lambda n=name: self._select_sign(n)
            )
            btn.pack(fill="x", pady=1, padx=2)
            self._sign_buttons[name] = btn

    def _on_search(self, *_):
        query = self.search_var.get().lower()
        matches = [s for s in self.signs if query in s.lower()]
        self._populate_list(matches)
        self.lbl_count.configure(text=f"{len(matches)} de {len(self.signs)} señas")

    def _select_sign(self, sign_name):
        if self._current_sign == sign_name:
            return
        # Highlight selected
        if self._current_sign and self._current_sign in self._sign_buttons:
            self._sign_buttons[self._current_sign].configure(fg_color="transparent")
        self._current_sign = sign_name
        if sign_name in self._sign_buttons:
            self._sign_buttons[sign_name].configure(fg_color="#1F6AA5")

        self._video_idx = 0
        self.lbl_sign.configure(text=sign_name.upper().replace("_", " "))
        self._play_current()

    def _play_current(self):
        self._stop_flag.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        self._stop_flag.clear()

        videos = self.signs.get(self._current_sign, [])
        if not videos:
            return
        self._video_idx = max(0, min(self._video_idx, len(videos) - 1))
        path = videos[self._video_idx]
        self.lbl_video_info.configure(
            text=f"Video {self._video_idx+1}/{len(videos)}: {os.path.basename(path)}"
        )
        self.lbl_nav.configure(text=f"← → para navegar entre {len(videos)} videos")

        self._play_thread = threading.Thread(
            target=self._video_loop, args=(path,), daemon=True
        )
        self._play_thread.start()

    def _video_loop(self, path):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps

        import time
        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                continue

            # Fit frame to canvas
            self.update_idletasks()
            cw = max(self.canvas.winfo_width(), 100)
            ch = max(self.canvas.winfo_height(), 100)
            h, w = frame.shape[:2]
            scale = min(cw / w, ch / h)
            nw, nh = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (nw, nh))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))

            def update_canvas(img=img, nw=nw, nh=nh, cw=cw, ch=ch):
                self._tk_image = img
                self.canvas.delete("all")
                self.canvas.create_image(cw//2, ch//2, anchor="center", image=img)

            try:
                self.after(0, update_canvas)
            except Exception:
                break
            time.sleep(delay)

        cap.release()

    def _prev_video(self):
        if not self._current_sign:
            return
        videos = self.signs.get(self._current_sign, [])
        self._video_idx = (self._video_idx - 1) % len(videos)
        self._play_current()

    def _next_video(self):
        if not self._current_sign:
            return
        videos = self.signs.get(self._current_sign, [])
        self._video_idx = (self._video_idx + 1) % len(videos)
        self._play_current()

    def _on_close(self):
        self._stop_flag.set()
        self.destroy()


if __name__ == "__main__":
    app = ReferenceViewer()
    app.mainloop()
