import os
import cv2
import csv
import yaml
import numpy as np
import mediapipe as mp
from torch import optim
from datetime import datetime
from torchmetrics import Accuracy
from torch.utils.data import Dataset

#Xác định các phím bấm hợp lệ để bắt đầu hoặc kết thúc việc ghi dữ liệu cho một class cử chỉ
def is_handsign_character(char:str):
    return ord('a') <= ord(char) <ord("q") or char == " "


#Đọc file hand_gesture.yaml và trả về một từ điển chứa các nhãn cử chỉ.
def label_dict_from_config_file(relative_path):
    with open(relative_path,"r") as f:
       label_tag = yaml.full_load(f)["gestures"]
    return label_tag

# • Chức năng: Ghi dữ liệu landmarks của bàn tay và nhãn tương ứng vào file CSV.
# • Method __init__: Khởi tạo và mở file CSV để ghi dữ liệu.
# • Method add: Thêm một dòng dữ liệu mới vào file CSV, bao gồm nhãn và tọa độ các landmarks.
# • Method close: Đóng file CSV sau khi hoàn tất ghi dữ liệu.

class HandDatasetWriter():
    def __init__(self,filepath) -> None:
        self.csv_file = open(filepath,"a")
        self.file_writer = csv.writer(self.csv_file,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    def add(self,hand,label):
        self.file_writer.writerow([label,*np.array(hand).flatten().tolist()])
    def close(self):
        self.csv_file.close()

# • Chức năng: Phát hiện bàn tay trong khung hình và trích xuất các landmarks.
# • Method __init__: Khởi tạo các thành phần cần thiết của MediaPipe để nhận diện bàn tay.
# • Method detectHand:
# – Chuyển đổi frame từ BGR sang RGB.
# – Sử dụng MediaPipe để nhận diện bàn tay và trích xuất các landmarks.
# – Vẽ các landmarks và kết nối lên hình ảnh để hiển thị.
# – Trả về danh sách các landmarks và hình ảnh đã vẽ.
class HandLandmarksDetector():
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(False,max_num_hands=1,min_detection_confidence=0.5)

    def detectHand(self,frame):
        hands = []
        frame = cv2.flip(frame, 1)
        annotated_image = frame.copy()
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for landmark in hand_landmarks.landmark:
                    x,y,z = landmark.x,landmark.y,landmark.z
                    hand.extend([x,y,z])
            hands.append(hand)
        return hands,annotated_image


def run(data_path, sign_img_path, split="val",resolution=(1280,720)):

    hand_detector = HandLandmarksDetector()
    cam =  cv2.VideoCapture(0)
    cam.set(3,resolution[0])
    cam.set(4,resolution[1])

    os.makedirs(data_path,exist_ok=True)
    os.makedirs(sign_img_path,exist_ok=True)
    print(sign_img_path)
    dataset_path = f"./{data_path}/landmark_{split}.csv"
    hand_dataset = HandDatasetWriter(dataset_path)
    current_letter= None
    status_text = None
    cannot_switch_char = False


    saved_frame = None
    while cam.isOpened():
        _,frame = cam.read()
        hands,annotated_image = hand_detector.detectHand(frame)
        
        if(current_letter is None):
            status_text = "press a character to record"
            
        else:
            label =  ord(current_letter)-ord("a")
            if label == -65:
                status_text = f"Recording unknown, press spacebar again to stop"
                label = -1
            else:
                status_text = f"Recording {LABEL_TAG[label]}, press {current_letter} again to stop"

        key = cv2.waitKey(1)
        if(key == -1):
            if(current_letter is None ):
                # no current letter recording, just skip it
                pass
            else:
                if len(hands) != 0:
                    hand = hands[0]
                    hand_dataset.add(hand=hand,label=label)
                    saved_frame = frame
        # some key is pressed
        else:
            # pressed some key, do not push this image, assign current letter to the key just pressed
            key = chr(key)
            if key == "q":
                break
            if (is_handsign_character(key)):
                if(current_letter is None):
                    current_letter = key
                elif(current_letter == key):
                    # pressed again?, reset the current state
                    if saved_frame is not None:
                        if label >=0:
                            cv2.imwrite(f"./{sign_img_path}/{LABEL_TAG[label]}.jpg",saved_frame)

                    cannot_switch_char=False
                    current_letter = None
                    saved_frame = None
                else:
                    cannot_switch_char = True
                    # warned user to unbind the current_letter first
        if(cannot_switch_char):
            cv2.putText(annotated_image, f"please press {current_letter} again to unbind", (0,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(annotated_image, status_text, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f"{split}",annotated_image)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    LABEL_TAG = label_dict_from_config_file("./Project/hand_gesture.yaml")
    data_path = './data2'
    sign_img_path = './sign_imgs2'
    run(data_path, sign_img_path, "train",(1280,720))
    run(data_path, sign_img_path, "val",(1280,720))
    run(data_path, sign_img_path, "test",(1280,720))
  
class NeuralNetwork(nn.Modulee):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file("hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(63, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),

                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.4),

                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.4),

                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.6),


                nn.Linear(128, num_classes)  # Replace num_classes with the actual number of classes for gesture recognition
            )
    def forward(self,x):
        x = self.flatten(x)
        logits - self.linear_relu_stack(x)
    
    def predict(self,x,threshold = 0.8):
        logits = self(x)
        softmax_prob = nn.Softmax(dim =1)(logits)
        chosen_ind = torch.argmax(softmax_prob,dim=1)
        return torch_where(softmax_prob[0,chosen_ind]>threshold,chosen_ind,-1)
    
    def predict_with_known_class(self,x):
        logits = self(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        return torch.argmax(softmax_prob,dim=1)
    
    def score(self,logits):
        return -torch.max(logits,dim=1)

def label_dict_from_config_file(relative_path):
    with open(relative_path,"r") as f:
       label_tag = yaml.full_load(f)["gestures"]
    return label_tag

class CustomImageDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.labels = torch.from_numpy(self.data.iloc[:,0].to_numpy())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_hot_label = self.labels[idx]
        torch_data = torch.from_numpy(self.data.iloc[idx,1:].to_numpy(dtype=np.float32))
        return torch_data, one_hot_label
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.watched_metrics = np.inf

    def early_stop(self, current_value):
        if current_value < self.watched_metrics:
            self.watched_metrics = current_value
            self.counter = 0
        elif current_value > (self.watched_metrics + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def train(trainloader, val_loader, model, loss_function, early_stopper, optimizer):
    # add auroc score
    best_vloss = 1_000_000
    timestamp = datetime.now().strftime('%d-%m %H:%M')
    for epoch in range(300):
        #training step
        model.train(True)
        running_loss = 0.0
        acc_train = Accuracy(num_classes=len(LIST_LABEL), task='MULTICLASS')
        for batch_number,data in enumerate(trainloader):
            inputs,labels = data
            optimizer.zero_grad()
            preds=model(inputs)
            loss = loss_function(preds,labels)
            loss.backward()
            optimizer.step()
            acc_train.update(model.predict_with_known_class(inputs), labels)
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        # validating step
        model.train(False)
        running_vloss = 0.0
        acc_val = Accuracy(num_classes=len(LIST_LABEL), task='MULTICLASS')
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            preds = model(vinputs)
            vloss = loss_function(preds, vlabels)
            running_vloss += vloss.item()
            acc_val.update(model.predict_with_known_class(vinputs), vlabels)

        # Log the running loss averaged per batch
        # for both training and validation
        print(f"Epoch {epoch}: ")
        print(f"Accuracy train:{acc_train.compute().item()}, val:{acc_val.compute().item()}")
        avg_vloss = running_vloss / len(val_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        print('Training vs. Validation accuracy',
                        { 'Training' : acc_train.compute().item()
                        , 'Validation' : acc_val.compute().item() },
                        epoch + 1)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_path = f'./{save_path}/model_{timestamp}_{model.__class__.__name__}_best'
            torch.save(model.state_dict(), best_model_path)

        if early_stopper.early_stop(avg_vloss):
            ################## Your Code Here ################## Q5
            ''' Hoàn thành đoạn code bên dướ để  print ra epoch hiện tại và
            minimum watched metric và thoát loop
            '''

            ####################################################



    model_path = f'./{save_path}/model_{timestamp}_{model.__class__.__name__}_last'
    torch.save(model.state_dict(), model_path)

    print(acc_val.compute())
    return model, best_model_path
DATA_FOLDER_PATH="./data/"
LIST_LABEL = label_dict_from_config_file("hand_gesture.yaml")
train_path = os.path.join(DATA_FOLDER_PATH,"landmark_train.csv")
val_path = os.path.join(DATA_FOLDER_PATH,"landmark_val.csv")
save_path = './models'
os.makedirs(save_path,exist_ok=True)

trainset = CustomImageDataset(train_path)
trainloader = torch.utils.data.DataLoadet(trainset,batch_size = 40,shuffle = True)



valset = CustomImageDataset(os.path.join(val_path))
val_loader = torch.utils.data.DataLoader(valset,batch_size=50, shuffle=False)

model = NeuralNetwork()
loss_function = mm.CrossEntropyLoss()
early_stopper = EaryStopper(patience = 30,min_delta =0.01)

optimizer =optim.Adam(model.parameters(),lr =0.0001)


model, best_model_path = train(trainloader, val_loader, model, loss_function, early_stopper, optimizer)

list_label = label_dict_from_config_file("hand_gesture.yaml")
DATA_FOLDER_PATH="./data/"
testset = CustomImageDataset(os.path.join(DATA_FOLDER_PATH,"landmark_test.csv"))
test_loader = torch.utils.data.DataLoader(testset,batch_size=20,shuffle=False)

network = NeuralNetwork()
network.load_state_dict(torch.load(best_model_path, weights_only=False))

network.eval()
acc_test = Accuracy(num_classes=len(list_label), task='MULTICLASS')
for i, test_data in enumerate(test_loader):
    test_input, test_label = test_data
    preds = network(test_input)
    acc_test.update(preds, test_label)
print(network.__class__.__name__)
print(f"Accuracy of model:{acc_test.compute().item()}")
