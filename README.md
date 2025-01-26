# YOLOv8-based-intelligent-contraband-detection-in-small-real-time-images
Project Process: A fast contraband detection system was designed using PMMW images for object detection. The project goals were defined, and challenges in resolution, contrast, and noise in PMMW images were analyzed. The PMMW image dataset was collected and preprocessed. YOLOv8 was chosen as the base model, and structural optimizations were made. GSConv was used in the Backbone to reduce redundant features, and GSConv-Slim was introduced in the Neck to maintain accuracy while reducing complexity. LADH was added to the detection head for efficient, accurate, and lightweight detection. The model was trained and evaluated for optimal performance.

Project Results: The model size was 2.3M with a compression rate of 26.98%, enabling lightweight deployment. This significantly improved safety detection efficiency in public places like airports and subway stations, effectively enhancing public safety.
<div style="text-align:center; margin-bottom: 20px;">
    <h3>Performance comparison of the model before and after quantization</h3>
    <img width="80%" alt="Performance comparison" src="https://github.com/user-attachments/assets/c2b6195e-85ad-4cd2-ab1a-cab4656bc391" />
</div>

<div style="text-align:center; margin-bottom: 20px;">
    <h3>Ablation study</h3>
    <img width="80%" alt="Ablation study" src="https://github.com/user-attachments/assets/56877732-a55e-40ed-84ad-46b357e19bbc" />
</div>

<div style="text-align:center; margin-bottom: 20px;">
    <h3>Before and after quantization</h3>
    <img width="80%" alt="Before and after quantization" src="https://github.com/user-attachments/assets/272872d6-0d20-4057-8c62-5f24a76d5115" />
</div>


