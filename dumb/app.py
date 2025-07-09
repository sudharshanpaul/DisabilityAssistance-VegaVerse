import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import tempfile
import os 
import torch.nn as nn
import torchvision.models as models
import json

# Set page config
st.set_page_config(
    page_title="Video Classification App",
    page_icon="🎬",
    layout="wide"
)

def get_top_k_labels(json_path, k=300):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Count instances for each gloss
        gloss_instance_counts = []
        for item in data:
            gloss = item.get("gloss")
            instances = item.get("instances", [])
            if gloss and isinstance(instances, list):
                gloss_instance_counts.append((gloss, len(instances)))

        # Sort by number of instances, descending
        gloss_instance_counts.sort(key=lambda x: x[1], reverse=True)

        # Take top-k
        top_k_glosses = [gloss for gloss, _ in gloss_instance_counts[:k]]

        # Create mappings
        class_labels = top_k_glosses
        label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # print(f"✅ Extracted top {k} class labels.")
        return class_labels, label_to_idx, idx_to_label

    except Exception as e:
        # print(f"❌ Error: {e}")
        return [], {}, {}
    

LABEL_JSON_PATH = r'C:\Users\Harsha PC\Desktop\deaf\WLASL_v0.3.json'
CLASS_LABELS, label_to_idx, idx_to_label = get_top_k_labels(LABEL_JSON_PATH)

print(len(CLASS_LABELS))

# Define your class labels here - update with your actual classes
# CLASS_LABELS = [
#     "book",
#     "drink", 
#     "computer",
#     "before",
#     "chair",
#     "go",
#     "clothes"



#     # Add more classes as needed
# ]


class SignLanguageModel(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=len(CLASS_LABELS)):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC
        self.rnn = nn.GRU(input_size=512, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        cnn_feats = []

        for t in range(T):
            out = self.cnn(x[:, t])  # (B, 512, 1, 1)
            out = out.view(B, -1)    # Flatten to (B, 512)
            cnn_feats.append(out)

        cnn_feats = torch.stack(cnn_feats, dim=1)  # (B, T, 512)
        _, h_n = self.rnn(cnn_feats)
        out = self.fc(h_n.squeeze(0))  # (B, num_classes)
        return out
    

@st.cache_resource
@st.cache_resource
def load_model(model_path):
    """Load the PyTorch model from saved state_dict"""
    try:
        # Define your model architecture
        model = SignLanguageModel(num_classes=len(CLASS_LABELS))  # Make sure CLASS_LABELS matches training
        state_dict = torch.load(model_path, map_location='cpu')  # Load the state_dict
        model.load_state_dict(state_dict)  # Load weights into the model
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_video(video_path, target_frames=16, target_size=(224, 224)):
    """
    Preprocess video to match model input requirements
    Expected output shape: (batch_size, channels, frames, height, width)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        st.error("No frames found in video")
        return None
    
    # Sample frames to get exactly target_frames
    if len(frames) >= target_frames:
        # Uniformly sample frames
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    else:
        # Repeat frames if we have fewer than target_frames
        while len(frames) < target_frames:
            frames.extend(frames[:target_frames - len(frames)])
    
    # Resize frames and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = []
    for frame in frames:
        processed_frame = transform(frame)
        processed_frames.append(processed_frame)
    
    # Stack frames: (frames, channels, height, width)
    video_tensor = torch.stack(processed_frames)
    
    # Rearrange to (channels, frames, height, width) and add batch dimension
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    
    return video_tensor

def predict_video(model, video_tensor):
    """Make prediction on preprocessed video"""
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def main():
    st.title("🎬 Video Classification App")
    st.markdown("Upload a video file to classify its content using your trained model.")
    
    # Sidebar for model configuration
    st.sidebar.header("Model Configuration")
    
    # Model upload
    model_file = st.sidebar.file_uploader(
        "Upload Model (.pth file)",
        type=['pth'],
        help="Upload your PyTorch model file"
    )
    
    if model_file is not None:
        # Save uploaded model temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_file.write(model_file.read())
            model_path = tmp_file.name
        
        # Load model
        model = load_model(model_path)
        
        if model is not None:
            st.sidebar.success("✅ Model loaded successfully!")
            
            # Video upload
            st.header("Upload Video")
            video_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for classification"
            )
            
            if video_file is not None:
                # Create two columns for layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📹 Input Video")
                    
                    # Save uploaded video temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                        tmp_video.write(video_file.read())
                        video_path = tmp_video.name
                    
                    # Display video
                    st.video(video_path)
                    
                    # Video info
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    
                    st.info(f"""
                    **Video Information:**
                    - Total Frames: {total_frames}
                    - FPS: {fps:.2f}
                    - Duration: {duration:.2f} seconds
                    """)
                
                with col2:
                    st.subheader("🎯 Prediction Results")
                    
                    if st.button("🚀 Classify Video", type="primary"):
                        with st.spinner("Processing video..."):
                            # Preprocess video
                            video_tensor = preprocess_video(video_path)
                            
                            if video_tensor is not None:
                                st.success(f"✅ Video preprocessed successfully!")
                                st.info(f"Tensor shape: {video_tensor.shape}")
                                
                                # Make prediction
                                try:
                                    predicted_class, confidence, all_probs = predict_video(model, video_tensor)
                                    
                                    # Display results
                                    st.success(f"🎉 Classification Complete!")
                                    
                                    # Predicted class
                                    class_name = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else f"Class {predicted_class}"
                                    st.metric(
                                        label="Predicted Class",
                                        value=class_name,
                                        delta=f"{confidence:.2%} confidence"
                                    )
                                    
                                    # Confidence bar
                                    st.progress(confidence)
                                    
                                    # All class probabilities
                                    st.subheader("📊 All Class Probabilities")
                                    for i, prob in enumerate(all_probs):
                                        class_name = CLASS_LABELS[i] if i < len(CLASS_LABELS) else f"Class {i}"
                                        st.write(f"{class_name}: {prob:.3f} ({prob*100:.1f}%)")
                                        st.progress(int(prob * 100))
                                    
                                except Exception as e:
                                    st.error(f"Error during prediction: {str(e)}")
                                    st.error("Make sure your model is compatible with the expected input shape.")
                
                # Clean up temporary files
                try:
                    os.unlink(video_path)
                except:
                    pass
        
        # Clean up temporary model file
        try:
            os.unlink(model_path)
        except:
            pass
    
    else:
        st.sidebar.warning("Please upload a model file to get started.")
        st.info("👈 Upload your .pth model file in the sidebar to begin video classification.")

    # Instructions
    st.markdown("---")
    st.markdown("""
    ## 📝 Instructions
    
    1. **Upload Model**: Use the sidebar to upload your trained PyTorch model (.pth file)
    2. **Upload Video**: Select a video file for classification
    3. **Classify**: Click the "Classify Video" button to get predictions
    
    ### 🔧 Model Requirements
    - Model should accept input shape: `(batch_size, 3, 16, 224, 224)`
    - Model should output class logits
    - Update the `CLASS_LABELS` list in the code with your actual class names
    
    ### 📹 Supported Video Formats
    - MP4, AVI, MOV, MKV
    
    ### ⚙️ Preprocessing Details
    - Videos are sampled to 16 frames
    - Frames are resized to 224x224 pixels
    - ImageNet normalization is applied
    - Tensor format: (batch_size, channels, frames, height, width)
    """)

if __name__ == "__main__":
    main()