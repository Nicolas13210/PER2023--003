import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from torchsummary import summary
#(240,424,3) --> RGB
#(480,848,1) --> BW


# # Load pre-trained ResNet-18 model
# mbnet_1 = models.mobilenet_v3_small(pretrained=True)
# mbnet_2 = models.mobilenet_v3_small(pretrained=True)
# # summary(mbnet_1,(3, 240, 424))
# # print(resnet_2)

# # Modify the first convolutional layer to accept grayscale images
# mbnet_2.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

# mbnet_1.classifier[3] = nn.Identity()
# mbnet_2.classifier[3] = nn.Identity()


# Define the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mbnet_1 = models.mobilenet_v3_small(pretrained=True)
        self.mbnet_2 = models.mobilenet_v3_small(pretrained=True)

        self.mbnet_2.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.mbnet_1.classifier[3] = nn.Identity()
        self.mbnet_2.classifier[3] = nn.Identity()

        self.fc_1 = nn.Linear(2048, 512) # Change the input size to reflect the modified output size

        self.fc1_steer = nn.Linear(512, 128)
        self.fc1_speed = nn.Linear(512, 128)

        self.fc2_steer = nn.Linear(128, 1)
        self.fc2_speed = nn.Linear(128, 1)

    def forward(self, x1, x2):
        # Pass the RGB image through the ResNet
        x1 = self.mbnet_1(x1)
        x1 = torch.flatten(x1, 1)

        # Pass the black and white image through the ResNet
        x2 = self.mbnet_2(x2)
        x2 = torch.flatten(x2, 1)

        # Concatenate the feature vectors
        x = torch.cat((x1, x2), dim=1)

        # Map the concatenated feature vector to a common size
        x = self.fc_1(x)

        # Predict steering and speed
        speed_predict = self.fc1_speed(x)
        steer_predict = self.fc1_steer(x)

        speed_predict = self.fc2_speed(speed_predict)
        steer_predict = self.fc2_steer(steer_predict)

        return torch.sigmoid(speed_predict), torch.sigmoid(steer_predict)

def get_mobilenet_1():
    # Create an instance of the model and move it to the GPU if available
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    # Output the summary of the model
    # summary(model, [(3, 240, 424), (1, 480, 848)])

    return model

class Model_classif_2(nn.Module):
    def __init__(self):
        super(Model_classif_2, self).__init__()
        
        self.deepwise_conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding='same', groups=3)
        self.pointwise_conv2 = torch.nn.Conv2d(3, 64, kernel_size=1, padding='same')
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.deepwise_conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, padding='same', groups=64)
        self.pointwise_conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, padding='same')
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)

        self.deepwise_conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, padding='same', groups=128)
        self.pointwise_conv6 = torch.nn.Conv2d(128, 256, kernel_size=1, padding='same')
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(256)

        self.deepwise_conv7 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
        self.pointwise_conv8 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bn8 = torch.nn.BatchNorm2d(256)

        self.deepwise_conv9 = torch.nn.Conv2d(256, 256, kernel_size=3, padding='same', groups=256)
        self.pointwise_conv10 = torch.nn.Conv2d(256, 256, kernel_size=1, padding='same')
        self.bn9 = torch.nn.BatchNorm2d(256)
        self.bn10 = torch.nn.BatchNorm2d(256)

        self.dense1 = torch.nn.Linear(19968, 512, bias=True)
        self.dense2 = torch.nn.Linear(512, 128, bias=True)
        self.dense3 = torch.nn.Linear(128, 1, bias=True)

        self.maxpooling = torch.nn.MaxPool2d(2, stride=2)
        self.maxpooling_half = torch.nn.MaxPool2d((1,2), stride=(1,2))
        self.relu = torch.nn.ReLU()

        self.dropout = torch.nn.Dropout(0.1)

    '''
    this function is made to compute prediction using the given batch
    args:
        x: torch tensor representing one batch of data
    
    return:
        x: torch tensor which contains a batch of prediction
    '''
    def forward(self, x):

        # x.shape == (512, 384)
        x = self.deepwise_conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.pointwise_conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (256, 192)
        x = self.deepwise_conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.pointwise_conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (128, 96)
        x = self.deepwise_conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.pointwise_conv6(x)
        x = self.relu(x)
        x = self.bn6(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (64, 48)
        x = self.deepwise_conv7(x)
        x = self.relu(x)
        x = self.bn7(x)
        x = self.pointwise_conv8(x)
        x = self.relu(x)
        x = self.bn8(x)
        x = self.maxpooling(x)
        x = self.dropout(x)

        # x.shape == (32, 24)
        x = self.deepwise_conv9(x)
        x = self.relu(x)
        x = self.bn9(x)
        x = self.pointwise_conv10(x)
        x = self.relu(x)
        x = self.bn10(x)
        x = self.maxpooling_half(x)
        x = self.dropout(x)

        # x.shape == (8, 6)
        x = torch.flatten(x, start_dim=1)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.dense3(x)

        return x
   

def get_mobilenet_classif_2():
    # Create an instance of the model and move it to the GPU if available
    model = Model_classif_2()
    if torch.cuda.is_available():
        model.cuda()

    # Output the summary of the model
    

    return model