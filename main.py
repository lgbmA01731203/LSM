import os
import sys
import time
import threading
import tkinter as tk
from tkinter import messagebox
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import torch
import utils

from camera_pipeline import LSMCameraPipeline
from data_collector import LSMDataCollector
from translator import LSMTranslator
from train_models import train_static_letters, train_dynamic_letters, train_words

# Set CTA appearance and theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class LSMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Traductor en Vivo de Lenguaje de Señas Mexicano (LSM)")
        self.geometry("1100x700")
        
        # Initialize components
        self.pipeline = LSMCameraPipeline()
        self.translator = LSMTranslator()
        
        # Data recording variables
        self.recording_active = False
        self.recording_sequence = []
        self.recording_label = ""
        self.recording_mode = "static"
        
        # Auto-spacing tracking
        self.last_hand_seen_time = time.time()
        self.space_appended = True

        # Cache widget dimensions to avoid calling winfo_width every frame
        self._cam_w = 0
        self._cam_h = 0
        self._rec_w = 0
        self._rec_h = 0

        # Create Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Tab View
        self.tabview = ctk.CTkTabview(self, width=1080, height=680)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.tab_translate = self.tabview.add("Traductor en Vivo")
        self.tab_record = self.tabview.add("Grabador de Señas")
        self.tab_train = self.tabview.add("Entrenador de Modelos")
        
        self.setup_translate_tab()
        self.setup_record_tab()
        self.setup_train_tab()
        
        # Start camera pipeline
        self.pipeline.start()
        
        # Begin UI update loop
        self.update_frame()
        
        # Handle close window
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def setup_translate_tab(self):
        # Configure grid for translation tab
        self.tab_translate.grid_columnconfigure(0, weight=3) # Camera area
        self.tab_translate.grid_columnconfigure(1, weight=1) # Settings and outputs
        self.tab_translate.grid_rowconfigure(0, weight=1)
        
        # Camera Feed Frame
        self.camera_frame = ctk.CTkFrame(self.tab_translate)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.camera_frame.grid_columnconfigure(0, weight=1)
        self.camera_frame.grid_rowconfigure(0, weight=1)
        
        self.video_label = tk.Label(self.camera_frame, text="Cargando Cámara...", font=("Inter", 16), bg="#2B2B2B", fg="white")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Sidebar Panel
        self.translate_sidebar = ctk.CTkFrame(self.tab_translate)
        self.translate_sidebar.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.translate_sidebar.grid_columnconfigure(0, weight=1)
        
        # Title
        self.lbl_side_title = ctk.CTkLabel(self.translate_sidebar, text="TRADUCCIÓN LSM", font=("Inter", 18, "bold"))
        self.lbl_side_title.grid(row=0, column=0, padx=10, pady=15)
        
        # Translation Mode Selector
        self.lbl_mode = ctk.CTkLabel(self.translate_sidebar, text="Modo de Traducción:", font=("Inter", 14))
        self.lbl_mode.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.mode_var = ctk.StringVar(value="static")
        self.mode_switch = ctk.CTkSegmentedButton(
            self.translate_sidebar, 
            values=["static", "dynamic"],
            command=self.change_translation_mode,
            variable=self.mode_var
        )
        self.mode_switch.configure(values=["Letras", "Palabras"])
        self.mode_switch.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        # Live Prediction Label
        self.lbl_prediction = ctk.CTkLabel(self.translate_sidebar, text="Seña Actual: ---", font=("Inter", 16, "bold"), text_color="#1F85DE")
        self.lbl_prediction.grid(row=3, column=0, padx=10, pady=15, sticky="w")
        
        # Hand Orientation Label
        self.lbl_orientation = ctk.CTkLabel(self.translate_sidebar, text="Orientación: ---", font=("Inter", 14, "bold"), text_color="#2ECC71")
        self.lbl_orientation.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        
        # Confidence progress bar
        self.lbl_confidence = ctk.CTkLabel(self.translate_sidebar, text="Confianza: 0%", font=("Inter", 12))
        self.lbl_confidence.grid(row=5, column=0, padx=10, pady=0, sticky="w")
        
        self.progress_conf = ctk.CTkProgressBar(self.translate_sidebar)
        self.progress_conf.set(0)
        self.progress_conf.grid(row=6, column=0, padx=10, pady=5, sticky="ew")
        
        # Translated Text Box
        self.lbl_out_text = ctk.CTkLabel(self.translate_sidebar, text="Texto Traducido:", font=("Inter", 14))
        self.lbl_out_text.grid(row=7, column=0, padx=10, pady=10, sticky="w")
        
        self.txt_translation = ctk.CTkTextbox(self.translate_sidebar, height=150, font=("Inter", 14))
        self.txt_translation.grid(row=8, column=0, padx=10, pady=5, sticky="ew")
        
        # Buttons
        self.btn_space = ctk.CTkButton(self.translate_sidebar, text="Espacio", fg_color="#4F4F4F", command=self.add_space)
        self.btn_space.grid(row=9, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_backspace = ctk.CTkButton(self.translate_sidebar, text="Borrar Último", fg_color="#C0392B", hover_color="#922B21", command=self.backspace_text)
        self.btn_backspace.grid(row=10, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_clear = ctk.CTkButton(self.translate_sidebar, text="Limpiar Todo", fg_color="#7F8C8D", hover_color="#5D6D7E", command=self.clear_text)
        self.btn_clear.grid(row=11, column=0, padx=10, pady=5, sticky="ew")
        
        self.btn_tts = ctk.CTkButton(self.translate_sidebar, text="Escuchar (TTS)", command=self.speak_translation)
        self.btn_tts.grid(row=12, column=0, padx=10, pady=5, sticky="ew")
        
        # Correction UI Frame
        self.correction_frame = ctk.CTkFrame(self.translate_sidebar)
        self.correction_frame.grid(row=13, column=0, padx=10, pady=10, sticky="ew")
        self.correction_frame.grid_columnconfigure(0, weight=2)
        self.correction_frame.grid_columnconfigure(1, weight=1)
        self.correction_frame.grid_columnconfigure(2, weight=1)
        
        self.lbl_corr_title = ctk.CTkLabel(self.correction_frame, text="Corregir seña a:", font=("Inter", 12))
        self.lbl_corr_title.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.entry_correct_char = ctk.CTkEntry(self.correction_frame, placeholder_text="ej. A", width=40)
        self.entry_correct_char.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.btn_correct = ctk.CTkButton(self.correction_frame, text="Aprender", width=80, fg_color="#E67E22", hover_color="#D35400", command=self.correct_current_prediction)
        self.btn_correct.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        # Status Label
        self.lbl_status = ctk.CTkLabel(self.translate_sidebar, text="Modelos: RF [No] | LSTM [No]", font=("Inter", 11), text_color="#E74C3C")
        self.lbl_status.grid(row=14, column=0, padx=10, pady=15, sticky="w")
        self.update_model_status_label()
        
    def setup_record_tab(self):
        # Grid configs
        self.tab_record.grid_columnconfigure(0, weight=3) # Video preview
        self.tab_record.grid_columnconfigure(1, weight=1) # Panel settings
        self.tab_record.grid_rowconfigure(0, weight=1)
        
        # Camera Feed view (reused inside pipeline)
        self.record_camera_frame = ctk.CTkFrame(self.tab_record)
        self.record_camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.record_camera_frame.grid_columnconfigure(0, weight=1)
        self.record_camera_frame.grid_rowconfigure(0, weight=1)
        
        self.record_video_label = tk.Label(self.record_camera_frame, text="Cargando Cámara...", font=("Inter", 16), bg="#2B2B2B", fg="white")
        self.record_video_label.grid(row=0, column=0, sticky="nsew")
        
        # Recorder configuration panel
        self.record_panel = ctk.CTkFrame(self.tab_record)
        self.record_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.record_panel.grid_columnconfigure(0, weight=1)
        
        # Title
        self.lbl_rec_title = ctk.CTkLabel(self.record_panel, text="GRABADOR DE GESTOS", font=("Inter", 18, "bold"))
        self.lbl_rec_title.grid(row=0, column=0, padx=10, pady=15)
        
        # Label Input
        self.lbl_rec_name = ctk.CTkLabel(self.record_panel, text="Nombre de la Seña / Etiqueta:", font=("Inter", 14))
        self.lbl_rec_name.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.entry_rec_label = ctk.CTkEntry(self.record_panel, placeholder_text="ej. 'A', 'hola', 'gracias'")
        self.entry_rec_label.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        
        # Seña type
        self.lbl_rec_mode = ctk.CTkLabel(self.record_panel, text="Tipo de Seña:", font=("Inter", 14))
        self.lbl_rec_mode.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        self.rec_mode_var = ctk.StringVar(value="static_letter")
        self.rec_mode_switch = ctk.CTkSegmentedButton(
            self.record_panel,
            values=["static_letter", "dynamic_letter", "word"],
            variable=self.rec_mode_var
        )
        self.rec_mode_switch.configure(values=["Letra Estática", "Letra Dinámica", "Palabra"])
        self.rec_mode_switch.grid(row=4, column=0, padx=10, pady=5, sticky="ew")
        
        # Counter of current signs recorded
        self.lbl_samples_count = ctk.CTkLabel(self.record_panel, text="Muestras Guardadas: 0", font=("Inter", 14, "bold"))
        self.lbl_samples_count.grid(row=5, column=0, padx=10, pady=20)
        
        # Record buttons
        self.btn_capture = ctk.CTkButton(self.record_panel, text="Grabar Muestra", fg_color="#2ECC71", hover_color="#27AE60", command=self.trigger_recording)
        self.btn_capture.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
        
        self.lbl_rec_instructions = ctk.CTkLabel(
            self.record_panel, 
            text="Instrucciones:\n1. Coloca tu mano en la cámara.\n2. Presiona 'Grabar Muestra'.\n3. Si es dinámica, realiza la seña\ndurante 1 segundo.",
            font=("Inter", 11),
            justify="left"
        )
        self.lbl_rec_instructions.grid(row=7, column=0, padx=10, pady=20, sticky="w")
        
    def setup_train_tab(self):
        # Configure layout for training
        self.tab_train.grid_columnconfigure(0, weight=1)
        self.tab_train.grid_rowconfigure(2, weight=1) # Log textbox gets extra height
        
        self.lbl_train_title = ctk.CTkLabel(self.tab_train, text="ENTRENADOR DE MODELOS LOCALES", font=("Inter", 18, "bold"))
        self.lbl_train_title.grid(row=0, column=0, padx=10, pady=15)
        
        # Buttons Frame
        self.train_buttons_frame = ctk.CTkFrame(self.tab_train)
        self.train_buttons_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.train_buttons_frame.grid_columnconfigure(0, weight=1)
        self.train_buttons_frame.grid_columnconfigure(1, weight=1)
        
        self.train_buttons_frame.grid_columnconfigure(2, weight=1)
        
        self.btn_train_static = ctk.CTkButton(
            self.train_buttons_frame, 
            text="Letras Estáticas (RF)", 
            command=lambda: self.run_training_thread("static_letters")
        )
        self.btn_train_static.grid(row=0, column=0, padx=10, pady=15, sticky="ew")
        
        self.btn_train_dyn_let = ctk.CTkButton(
            self.train_buttons_frame, 
            text="Letras Dinámicas (LSTM)", 
            command=lambda: self.run_training_thread("dynamic_letters")
        )
        self.btn_train_dyn_let.grid(row=0, column=1, padx=10, pady=15, sticky="ew")
        
        self.btn_train_words = ctk.CTkButton(
            self.train_buttons_frame, 
            text="Palabras (LSTM)", 
            command=lambda: self.run_training_thread("words")
        )
        self.btn_train_words.grid(row=0, column=2, padx=10, pady=15, sticky="ew")
        
        # Console output for logs
        self.lbl_logs_title = ctk.CTkLabel(self.tab_train, text="Consola de Entrenamiento:", font=("Inter", 14))
        self.lbl_logs_title.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.txt_train_logs = ctk.CTkTextbox(self.tab_train, height=350, font=("Consolas", 12))
        self.txt_train_logs.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        self.txt_train_logs.insert("0.0", "Haz clic en uno de los botones para entrenar con tus datos grabados...\n")
        self.txt_train_logs.configure(state="disabled")

    def change_translation_mode(self, mode_alias):
        # Mapping alias
        if "Letras" in mode_alias:
            self.translator.no_hand_counter = 0
            self.translator.last_committed_letter = None
            self.translator.letter_history.clear()
            self.mode_var.set("static")
        else:
            self.translator.dynamic_cooldown = 0
            self.translator.sequence.clear()
            self.mode_var.set("dynamic")
            
    def update_model_status_label(self):
        has_static = "Sí" if self.translator.static_classifier is not None else "No"
        has_dyn_let = "Sí" if self.translator.dynamic_letters_model is not None else "No"
        has_words = "Sí" if self.translator.words_model is not None else "No"
        color = "#2ECC71" if (has_static == "Sí" or has_dyn_let == "Sí" or has_words == "Sí") else "#E74C3C"
        
        # Check CUDA
        cuda_status = "CUDA" if torch.cuda.is_available() else "CPU"
        
        self.lbl_status.configure(
            text=f"Modelos: L. Estáticas [{has_static}] | L. Dinámicas [{has_dyn_let}] | Palabras [{has_words}] | {cuda_status}",
            text_color=color
        )

    def trigger_recording(self):
        label = self.entry_rec_label.get().strip().lower()
        if not label:
            messagebox.showerror("Error", "Debes ingresar una etiqueta/nombre para la seña.")
            return
            
        mode = self.rec_mode_var.get()
        self.recording_label = label
        self.recording_mode = mode
        
        collector = LSMDataCollector(mode=mode, label=label)
        
        if mode == 'static_letter':
            # Record static sample immediately from the next feature frame
            # Handled in the update loop
            self.recording_active = True
        else:
            # Trigger dynamic sequence capture
            self.recording_sequence = []
            self.recording_active = True
            self.btn_capture.configure(state="disabled", text="Grabando...")
            
    def update_frame(self):
        try:
            # 1. Fetch latest data from camera pipeline
            frame, features = self.pipeline.get_latest()
            
            # 2. Run real-time translation if data is ready
            if features is not None:
                active_mode_raw = self.mode_var.get()
                if active_mode_raw in ['Letras', 'static', 'Letra Estática']:
                    active_mode = 'static'
                else:
                    active_mode = 'dynamic'
                
                # Check for hands present (at least one non-zero coordinate)
                hands_present = not np.all(features == 0)
                
                if hands_present:
                    self.last_hand_seen_time = time.time()
                    self.space_appended = False
                else:
                    # Auto-spacing logic: if hand is missing for > 2 seconds in static mode, append a space
                    if active_mode == 'static' and not self.space_appended:
                        if time.time() - self.last_hand_seen_time > 2.0:
                            self.add_space()
                            self.space_appended = True
                
                # Run inference
                new_token, raw_prediction, confidence = self.translator.process_frame(features, mode=active_mode)
                
                # Get hand orientations
                left_orient, right_orient = utils.get_hand_orientation(features)
                orient_strs = []
                if left_orient:
                    orient_strs.append(f"Izq: {left_orient.upper()}")
                if right_orient:
                    orient_strs.append(f"Der: {right_orient.upper()}")
                
                if orient_strs:
                    self.lbl_orientation.configure(text=f"Orientación: {', '.join(orient_strs)}")
                else:
                    self.lbl_orientation.configure(text="Orientación: ---")
                
                # Update GUI elements
                if raw_prediction:
                    if confidence >= 0.55:
                        self.lbl_prediction.configure(text=f"Seña Actual: {raw_prediction.upper()}")
                    else:
                        self.lbl_prediction.configure(text="Seña Actual: ---")
                    self.progress_conf.set(confidence)
                    self.lbl_confidence.configure(text=f"Confianza: {confidence * 100:.1f}%")
                else:
                    self.lbl_prediction.configure(text="Seña Actual: ---")
                    self.progress_conf.set(0)
                    self.lbl_confidence.configure(text="Confianza: 0%")
                    if not hands_present:
                        self.lbl_prediction.configure(text="Seña Actual: --- (Sin mano)")
                        self.lbl_orientation.configure(text="Orientación: --- (Sin mano)")
                        
                if new_token:
                    # If dynamic, append with space, if static, append directly
                    if active_mode == 'dynamic':
                        self.txt_translation.insert(tk.END, new_token + " ")
                    else:
                        self.txt_translation.insert(tk.END, new_token)
                    self.txt_translation.see(tk.END)
                    
                # 3. Handle data collector recording logic
                if self.recording_active:
                    if self.recording_mode == 'static_letter':
                        if hands_present:
                            collector = LSMDataCollector(mode='static_letter', label=self.recording_label)
                            filepath = collector.save_static_sample(features)
                            self.recording_active = False
                            self.update_record_count_label()
                            messagebox.showinfo("Éxito", f"Muestra estática guardada!")
                        else:
                            # Wait until a hand is in frame
                            self.btn_capture.configure(text="Esperando mano...")
                    else:
                        # Dynamic mode recording (dynamic_letter or word)
                        if hands_present:
                            self.recording_sequence.append(features)
                            # Show progress on capture button
                            self.btn_capture.configure(text=f"Secuencia: {len(self.recording_sequence)}/30")
                            
                            if len(self.recording_sequence) >= 30:
                                collector = LSMDataCollector(mode=self.recording_mode, label=self.recording_label)
                                filepath = collector.save_dynamic_sample(self.recording_sequence)
                                self.recording_active = False
                                self.recording_sequence = []
                                self.btn_capture.configure(state="normal", text="Grabar Muestra")
                                self.update_record_count_label()
                                messagebox.showinfo("Éxito", "Secuencia de 30 cuadros guardada!")
                        else:
                            # Skip this frame or display warning on button
                            self.btn_capture.configure(text="Esperando mano...")
                elif not self.recording_active and self.btn_capture.cget("text") != "Grabar Muestra":
                    self.btn_capture.configure(state="normal", text="Grabar Muestra")
                    
            # 4. Render frames to TK UI
            if frame is not None:
                active_tab = self.tabview.get()

                if active_tab == "Traductor en Vivo":
                    # Refresh cached size occasionally (every resize event is caught by winfo)
                    w = self.camera_frame.winfo_width()
                    h = self.camera_frame.winfo_height()
                    if w > 10 and h > 10:
                        if w != self._cam_w or h != self._cam_h:
                            self._cam_w, self._cam_h = w, h
                        # Fast OpenCV resize on numpy array
                        resized = cv2.resize(frame, (self._cam_w, self._cam_h), interpolation=cv2.INTER_LINEAR)
                        img_pil = Image.fromarray(resized)
                        img_tk = ImageTk.PhotoImage(image=img_pil)
                        self.video_label.configure(image=img_tk, text="")
                        self.video_label.image = img_tk
                elif active_tab == "Grabador de Señas":
                    w = self.record_camera_frame.winfo_width()
                    h = self.record_camera_frame.winfo_height()
                    if w > 10 and h > 10:
                        if w != self._rec_w or h != self._rec_h:
                            self._rec_w, self._rec_h = w, h
                        # Fast OpenCV resize on numpy array
                        resized = cv2.resize(frame, (self._rec_w, self._rec_h), interpolation=cv2.INTER_LINEAR)
                        img_pil = Image.fromarray(resized)
                        img_tk = ImageTk.PhotoImage(image=img_pil)
                        self.record_video_label.configure(image=img_tk, text="")
                        self.record_video_label.image = img_tk
        except Exception as e:
            import traceback
            with open("gui_error.log", "w") as log_file:
                log_file.write("Exception inside update_frame:\n")
                traceback.print_exc(file=log_file)
            print(f"GUI Thread Error: {e}")

        # 30fps UI refresh (~33ms) — enough for smooth video, much lighter than 60fps
        self.after(33, self.update_frame)

    def update_record_count_label(self):
        label = self.entry_rec_label.get().strip().lower()
        if not label:
            self.lbl_samples_count.configure(text="Muestras Guardadas: 0")
            return
            
        mode = self.rec_mode_var.get()
        collector = LSMDataCollector(mode=mode, label=label)
        class_dir = collector.get_class_dir()
        count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        self.lbl_samples_count.configure(text=f"Muestras Guardadas: {count}")
        
    def add_space(self):
        self.txt_translation.insert(tk.END, " ")
        self.txt_translation.see(tk.END)
        
    def backspace_text(self):
        # Remove last character
        current_text = self.txt_translation.get("1.0", tk.END)
        if len(current_text) > 1:
            self.txt_translation.delete("end-2c", "end-1c")
            
    def clear_text(self):
        self.txt_translation.delete("1.0", tk.END)
        self.translator.last_committed_letter = None
        
    def speak_translation(self):
        text = self.txt_translation.get("1.0", tk.END).strip()
        if not text:
            return
            
        def speak():
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                speaker.Speak(text)
            except Exception:
                # Fallback to pyttsx3 if loaded
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")
                    
        # Run SAPI Voice in thread to prevent GUI freezing
        threading.Thread(target=speak, daemon=True).start()

    def correct_current_prediction(self):
        # 1. Get the correct letter character entered by the user
        correct_char = self.entry_correct_char.get().strip().lower()
        if not correct_char or len(correct_char) != 1 or not correct_char.isalpha():
            messagebox.showerror("Error", "Debes ingresar una sola letra (A-Z).")
            return
            
        # 2. Fetch the latest features from the camera pipeline
        _, features = self.pipeline.get_latest()
        if features is None or np.all(features == 0):
            messagebox.showerror("Error", "No se detecta ninguna mano en la cámara para aprender.")
            return
            
        # 3. Save the keypoint coordinates to the corresponding static_letters folder
        dest_dir = os.path.join("data", "static_letters", correct_char)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Save filename with timestamp to avoid name collisions
        timestamp = int(time.time() * 1000)
        filepath = os.path.join(dest_dir, f"user_correction_{timestamp}.npy")
        np.save(filepath, features)
        
        # 4. Trigger training in background
        print(f"\n[Corrección] Guardado ejemplo para letra '{correct_char.upper()}' en {filepath}")
        self.run_training_thread("static_letters")
        
        # Clear the entry field and alert user
        self.entry_correct_char.delete(0, tk.END)
        messagebox.showinfo("Éxito", f"¡Guardado ejemplo para '{correct_char.upper()}'! Iniciando reentrenamiento...")

    def run_training_thread(self, mode):
        self.txt_train_logs.configure(state="normal")
        self.txt_train_logs.delete("1.0", tk.END)
        self.txt_train_logs.insert(tk.END, f"Iniciando entrenamiento del clasificador {mode.upper()}...\n")
        self.txt_train_logs.configure(state="disabled")
        
        self.btn_train_static.configure(state="disabled")
        self.btn_train_dyn_let.configure(state="disabled")
        self.btn_train_words.configure(state="disabled")
        
        # Intercept output print calls to show logs inside CTkTextbox
        class TextRedirector(object):
            def __init__(self, widget):
                self.widget = widget
            def write(self, str):
                self.widget.configure(state="normal")
                self.widget.insert(tk.END, str)
                self.widget.see(tk.END)
                self.widget.configure(state="disabled")
            def flush(self):
                pass
                
        def train_task():
            old_stdout = sys.stdout
            sys.stdout = TextRedirector(self.txt_train_logs)
            
            try:
                if mode == "static_letters":
                    success = train_static_letters()
                elif mode == "dynamic_letters":
                    success = train_dynamic_letters()
                else:
                    success = train_words()
                
                if success:
                    # Reload models
                    self.translator.load_models()
                    self.update_model_status_label()
                    print("\n[Éxito] Entrenamiento finalizado y modelos cargados!")
                else:
                    print("\n[Error] El entrenamiento falló. Revisa que tengas suficientes datos grabados.")
            except Exception as e:
                print(f"\n[Fallo Crítico] Ocurrió una excepción durante el entrenamiento:\n{e}")
            finally:
                sys.stdout = old_stdout
                self.btn_train_static.configure(state="normal")
                self.btn_train_dyn_let.configure(state="normal")
                self.btn_train_words.configure(state="normal")
                
        threading.Thread(target=train_task, daemon=True).start()

    def on_close(self):
        # Stop threads before closing
        self.pipeline.stop()
        self.destroy()

if __name__ == "__main__":
    # Workaround for high DPI screens
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
        
    app = LSMApp()
    app.mainloop()
