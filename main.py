import cv2 as cv
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilis import preprocess_crop
import torch.nn as nn

class CatModel(nn.Module):
    def __init__(self, input_channels=3, hidden_unit=16, output=2):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_unit, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_unit * 32 * 32, output)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

# Load trained weights on CPU
device = torch.device("cpu")
model = CatModel()
model.load_state_dict(torch.load("catmodel_weights.pth", map_location=device))
model.eval()

classes=["Cataract", "Normal"]
result_text= ""

base_options=python.BaseOptions(model_asset_path='face_landmarker.task')
options=vision.FaceLandmarkerOptions(base_options=base_options)
detector=vision.FaceLandmarker.create_from_options(options)

capture=cv.VideoCapture(0)

left_crop= np.array([])
right_crop= np.array([])

while True:
    istrue,frame=capture.read()
    if not istrue:
        break

    frame=cv.flip(frame, 1)
    h, w, _=frame.shape

    rgb_frame=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result=detector.detect(mp_image)

    if result.face_landmarks:
        landmark=result.face_landmarks[0]

        l1=min(int(landmark[469].x * w),int(landmark[471].x * w))
        l2=min(int(landmark[470].y * h),int(landmark[472].y * h))
        l3=max(int(landmark[471].x * w),int(landmark[469].x * w))
        l4=max(int(landmark[472].y * h),int(landmark[470].y * h))

        r1=min(int(landmark[474].x * w),int(landmark[476].x * w))
        r2=min(int(landmark[475].y * h),int(landmark[477].y * h))
        r3=max(int(landmark[476].x * w),int(landmark[474].x * w))
        r4=max(int(landmark[477].y * h),int(landmark[475].y * h))

        pad_l_x = int((l3 - l1) * 0.6)
        pad_l_y = int((l4 - l2) * 0.6)
        pad_r_x = int((r3 - r1) * 0.6)
        pad_r_y = int((r4 - r2) * 0.6)

        l1, l2 = max(0, l1 - pad_l_x), max(0, l2 - pad_l_y)
        l3, l4 = min(w, l3 + pad_l_x), min(h, l4 + pad_l_y)
        r1, r2 = max(0, r1 - pad_r_x), max(0, r2 - pad_r_y)
        r3, r4 = min(w, r3 + pad_r_x), min(h, r4 + pad_r_y)

        left_crop = frame[l2:l4, l1:l3]
        right_crop = frame[r2:r4, r1:r3]

        if not result_text:
            cv.putText(frame, "PRESS 'S' TO SCAN", (50, 50), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    if result_text:
        cv.putText(frame, result_text, (40, 80), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow("Webcam", frame)

    key=cv.waitKey(20)&0xFF
    if key==ord('q'):
        break
    if key==ord('s') and left_crop.size > 0 and right_crop.size > 0:
        left_tensor=preprocess_crop(left_crop)
        right_tensor=preprocess_crop(right_crop)

        with torch.no_grad():
            l_logits=model(left_tensor)
            r_logits=model(right_tensor)


            l_probs=torch.softmax(l_logits, dim=1)[0]
            r_probs=torch.softmax(r_logits, dim=1)[0]

           
            l_idx=torch.argmax(l_probs).item()
            r_idx =torch.argmax(r_probs).item()

            l_conf =l_probs[l_idx].item() * 100
            r_conf= r_probs[r_idx].item() * 100

            l_pred=classes[l_idx]
            r_pred=classes[r_idx]

       
        result_text = f"L: {l_pred} ({l_conf:.1f}%) | R: {r_pred} ({r_conf:.1f}%)"
        print(f"Prediction -> {result_text}")
        print("\n========== CATARACT DETECTION ==========")
        print("Model       : CatModel (TinyVGG)")
        print("Train       : 840 samples")
        print("Test        : 210 samples")
        print("Epochs      : 20")
        print("Train Acc   : 96.37%")
        print("Test Acc    : 91.58%")
        print("Made by     : me ")
        print("========================================\n")
capture.release()
cv.destroyAllWindows()