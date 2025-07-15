import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CIA Face Recognition – DevPat Edition")
        self.root.geometry("1000x800")
        
        # Initialize variables
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_folder = 'known_faces'
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
        self.load_known_faces()
        
    def setup_ui(self):
        # Main title
        title_label = tk.Label(self.root, text="CIA Face Recognition System", 
                              font=("Arial", 20, "bold"), fg="navy")
        title_label.pack(pady=10)
        
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for controls
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Known faces management
        known_faces_label = tk.Label(control_frame, text="Known Faces Management", 
                                    font=("Arial", 12, "bold"))
        known_faces_label.pack(pady=(0, 10))
        
        self.add_face_btn = ttk.Button(control_frame, text="Add New Face", 
                                      command=self.add_new_face)
        self.add_face_btn.pack(pady=5, fill=tk.X)
        
        self.refresh_faces_btn = ttk.Button(control_frame, text="Refresh Known Faces", 
                                           command=self.load_known_faces)
        self.refresh_faces_btn.pack(pady=5, fill=tk.X)
        
        self.test_face_btn = ttk.Button(control_frame, text="Test Face Detection", 
                                       command=self.test_face_detection)
        self.test_face_btn.pack(pady=5, fill=tk.X)
        
        # Known faces listbox
        self.faces_listbox = tk.Listbox(control_frame, height=6)
        self.faces_listbox.pack(pady=5, fill=tk.X)
        
        # Recognition settings
        settings_label = tk.Label(control_frame, text="Recognition Settings", 
                                 font=("Arial", 12, "bold"))
        settings_label.pack(pady=(10, 5))
        
        # Tolerance setting
        tolerance_label = tk.Label(control_frame, text="Tolerance (0.3-0.8):")
        tolerance_label.pack(pady=(5, 0))
        
        self.tolerance_var = tk.DoubleVar(value=0.6)
        self.tolerance_scale = tk.Scale(control_frame, from_=0.3, to=0.8, 
                                       resolution=0.05, orient=tk.HORIZONTAL,
                                       variable=self.tolerance_var)
        self.tolerance_scale.pack(fill=tk.X, pady=(0, 5))
        
        # Model selection
        model_label = tk.Label(control_frame, text="Detection Model:")
        model_label.pack(pady=(5, 0))
        
        self.model_var = tk.StringVar(value="hog")
        model_frame = tk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Radiobutton(model_frame, text="HOG (Fast)", variable=self.model_var, 
                      value="hog").pack(side=tk.LEFT)
        tk.Radiobutton(model_frame, text="CNN (Accurate)", variable=self.model_var, 
                      value="cnn").pack(side=tk.LEFT)
        
        # Separator
        separator = ttk.Separator(control_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)
        
        # Image upload and recognition
        upload_label = tk.Label(control_frame, text="Face Recognition", 
                               font=("Arial", 12, "bold"))
        upload_label.pack(pady=(0, 10))
        
        self.upload_btn = ttk.Button(control_frame, text="Upload Image", 
                                    command=self.upload_image)
        self.upload_btn.pack(pady=5, fill=tk.X)
        
        self.analyze_btn = ttk.Button(control_frame, text="Analyze Faces", 
                                     command=self.analyze_faces, state=tk.DISABLED)
        self.analyze_btn.pack(pady=5, fill=tk.X)
        
        self.clear_btn = ttk.Button(control_frame, text="Clear Results", 
                                   command=self.clear_results)
        self.clear_btn.pack(pady=5, fill=tk.X)
        
        # Right panel for image display
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image display area
        self.image_label = tk.Label(image_frame, text="No image uploaded", 
                                   bg="lightgray", width=50, height=25)
        self.image_label.pack(pady=5, fill=tk.BOTH, expand=True)
        
        # Log area
        log_label = tk.Label(self.root, text="Activity Log", font=("Arial", 12, "bold"))
        log_label.pack(pady=(10, 0))
        
        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=5, padx=20, fill=tk.X)
        
        self.log_text = tk.Text(log_frame, height=10, width=100)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def preprocess_image(self, image_path):
        """Preprocess image for better face detection"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if image is too large (keep aspect ratio)
        height, width = image_rgb.shape[:2]
        if width > 1000 or height > 1000:
            scale = min(1000/width, 1000/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))
        
        # Enhance contrast
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Ensure correct format
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        
        if not image_rgb.flags.c_contiguous:
            image_rgb = np.ascontiguousarray(image_rgb)
        
        return image_rgb

    def load_known_faces(self):
        """Load and encode known faces from the known_faces folder"""
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.faces_listbox.delete(0, tk.END)
        
        if not os.path.exists(self.face_folder):
            os.makedirs(self.face_folder)
            self.log(f"Created folder '{self.face_folder}'. Add face images to this folder.")
            return
        
        loaded_count = 0
        files = [f for f in os.listdir(self.face_folder) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.bmp'))]
        
        for filename in files:
            path = os.path.join(self.face_folder, filename)
            self.log(f"🔄 Processing {filename}...")
            
            try:
                # Preprocess image
                image_rgb = self.preprocess_image(path)
                if image_rgb is None:
                    self.log(f"❌ Couldn't read {filename}")
                    continue
                
                # Try multiple face detection approaches
                encodings = None
                
                # Try with HOG first (faster)
                try:
                    face_locations = face_recognition.face_locations(image_rgb, model="hog")
                    if face_locations:
                        encodings = face_recognition.face_encodings(image_rgb, face_locations)
                        self.log(f"  📍 HOG detected {len(face_locations)} face(s)")
                except:
                    pass
                
                # If HOG failed, try CNN (more accurate)
                if not encodings:
                    try:
                        face_locations = face_recognition.face_locations(image_rgb, model="cnn")
                        if face_locations:
                            encodings = face_recognition.face_encodings(image_rgb, face_locations)
                            self.log(f"  📍 CNN detected {len(face_locations)} face(s)")
                    except:
                        pass
                
                # If still no faces, try with different upsampling
                if not encodings:
                    try:
                        face_locations = face_recognition.face_locations(image_rgb, number_of_times_to_upsample=2)
                        if face_locations:
                            encodings = face_recognition.face_encodings(image_rgb, face_locations)
                            self.log(f"  📍 Upsampled detection found {len(face_locations)} face(s)")
                    except:
                        pass
                
                if encodings and len(encodings) > 0:
                    # Use the first (and usually best) face encoding
                    self.known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                    self.faces_listbox.insert(tk.END, name)
                    loaded_count += 1
                    self.log(f"✅ Successfully loaded: {name}")
                else:
                    self.log(f"🚫 No face found in {filename}")
                    
            except Exception as e:
                self.log(f"❌ Error processing {filename}: {str(e)}")
                continue
        
        self.log(f"📊 Total faces loaded: {loaded_count}")

    def test_face_detection(self):
        """Test face detection on a selected image"""
        file_path = filedialog.askopenfilename(
            title="Select image to test face detection",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.jfif *.bmp")]
        )
        
        if not file_path:
            return
        
        self.log(f"🧪 Testing face detection on: {os.path.basename(file_path)}")
        
        try:
            image_rgb = self.preprocess_image(file_path)
            if image_rgb is None:
                self.log("❌ Could not load image")
                return
            
            # Test different detection methods
            methods = [
                ("HOG (default)", lambda: face_recognition.face_locations(image_rgb, model="hog")),
                ("CNN", lambda: face_recognition.face_locations(image_rgb, model="cnn")),
                ("HOG + Upsample", lambda: face_recognition.face_locations(image_rgb, number_of_times_to_upsample=2, model="hog")),
                ("CNN + Upsample", lambda: face_recognition.face_locations(image_rgb, number_of_times_to_upsample=2, model="cnn"))
            ]
            
            for method_name, method_func in methods:
                try:
                    locations = method_func()
                    self.log(f"  {method_name}: {len(locations)} face(s) detected")
                except Exception as e:
                    self.log(f"  {method_name}: Error - {str(e)}")
                    
        except Exception as e:
            self.log(f"❌ Test failed: {str(e)}")

    def add_new_face(self):
        """Add a new face to the known faces database"""
        file_path = filedialog.askopenfilename(
            title="Select face image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.jfif *.bmp")]
        )
        
        if not file_path:
            return
        
        # Ask for person's name
        name = tk.simpledialog.askstring("Person Name", "Enter the person's name:")
        if not name:
            return
        
        try:
            # Preprocess and validate image
            image_rgb = self.preprocess_image(file_path)
            if image_rgb is None:
                messagebox.showerror("Error", "Could not read the image file.")
                return
            
            # Test face detection
            face_locations = face_recognition.face_locations(image_rgb)
            if not face_locations:
                # Try with CNN if HOG failed
                face_locations = face_recognition.face_locations(image_rgb, model="cnn")
            
            if not face_locations:
                messagebox.showerror("Error", "No face found in the image. Try a clearer image.")
                return
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image_rgb, face_locations)
            if not encodings:
                messagebox.showerror("Error", "Could not encode the face.")
                return
            
            # Save original image to known_faces folder
            image_bgr = cv2.imread(file_path)
            filename = f"{name}.jpg"
            save_path = os.path.join(self.face_folder, filename)
            cv2.imwrite(save_path, image_bgr)
            
            self.log(f"✅ Added new face: {name}")
            self.load_known_faces()  # Refresh the known faces
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add face: {str(e)}")
            self.log(f"❌ Error adding face: {str(e)}")

    def upload_image(self):
        """Upload an image for face recognition"""
        file_path = filedialog.askopenfilename(
            title="Select image for face recognition",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.jfif *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            # Load and display image
            self.current_image_path = file_path
            image = Image.open(file_path)
            
            # Resize image to fit display area
            display_size = (600, 500)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            # Load image for processing
            self.current_image = cv2.imread(file_path)
            
            self.analyze_btn.config(state=tk.NORMAL)
            self.log(f"📷 Image uploaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def analyze_faces(self):
        """Analyze faces in the uploaded image"""
        if self.current_image is None:
            messagebox.showerror("Error", "No image uploaded.")
            return
        
        if not self.known_face_encodings:
            messagebox.showerror("Error", "No known faces loaded. Please add some faces first.")
            return
        
        self.log("🔍 Starting face analysis...")
        
        # Run analysis in a separate thread to prevent UI freezing
        threading.Thread(target=self._perform_analysis, daemon=True).start()

    def _perform_analysis(self):
        """Perform the actual face recognition analysis"""
        try:
            # Preprocess the image
            image_rgb = self.preprocess_image(self.current_image_path)
            if image_rgb is None:
                self.log("❌ Could not preprocess image")
                return
            
            # Get settings
            tolerance = self.tolerance_var.get()
            model = self.model_var.get()
            
            self.log(f"🔧 Using model: {model.upper()}, tolerance: {tolerance}")
            
            # Find faces with selected model
            face_locations = face_recognition.face_locations(image_rgb, model=model)
            
            # If no faces found, try the other model
            if not face_locations:
                other_model = "cnn" if model == "hog" else "hog"
                self.log(f"🔄 No faces found with {model.upper()}, trying {other_model.upper()}...")
                face_locations = face_recognition.face_locations(image_rgb, model=other_model)
            
            # If still no faces, try with upsampling
            if not face_locations:
                self.log("🔄 Trying with upsampling...")
                face_locations = face_recognition.face_locations(image_rgb, number_of_times_to_upsample=2)
            
            self.log(f"👤 Found {len(face_locations)} face(s) in the image")
            
            if not face_locations:
                self.log("❌ No faces detected in the image")
                return
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
            
            # Create a copy of the original image for drawing
            result_image = self.current_image.copy()
            
            # Analyze each face
            for i, (face_encoding, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                name = "Unknown"
                confidence = 0
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                    
                    # Log all distances for debugging
                    for j, (known_name, distance) in enumerate(zip(self.known_face_names, face_distances)):
                        match_status = "✅" if matches[j] else "❌"
                        self.log(f"  {match_status} {known_name}: distance={distance:.3f}")
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                thickness = 3
                
                cv2.rectangle(result_image, (left, top), (right, bottom), color, thickness)
                
                # Draw label background
                label_height = 35
                cv2.rectangle(result_image, (left, bottom - label_height), (right, bottom), color, cv2.FILLED)
                
                # Draw label text
                label = f"{name}"
                if name != "Unknown":
                    label += f" ({confidence:.2f})"
                
                cv2.putText(result_image, label, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Log result
                if name != "Unknown":
                    self.log(f"✅ Face {i+1}: {name} (confidence: {confidence:.3f})")
                else:
                    self.log(f"❓ Face {i+1}: Unknown person (best distance: {min(face_distances):.3f})")
            
            # Display result image
            self._display_result_image(result_image)
            
        except Exception as e:
            self.log(f"❌ Analysis failed: {str(e)}")
            import traceback
            self.log(f"📋 Full error: {traceback.format_exc()}")

    def _display_result_image(self, cv_image):
        """Display the result image with face recognition annotations"""
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Resize to fit display area
        display_size = (600, 500)
        image_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image_pil)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
        self.log("🎯 Face recognition analysis completed!")

    def clear_results(self):
        """Clear the current image and results"""
        self.current_image = None
        self.current_image_path = None
        self.image_label.configure(image="", text="No image uploaded")
        self.image_label.image = None
        self.analyze_btn.config(state=tk.DISABLED)
        self.log("🧹 Results cleared")

# Import for dialog box
import tkinter.simpledialog

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()