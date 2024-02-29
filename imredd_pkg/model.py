import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
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
        self.mbnet_1 = models.mobilenet_v3_small(weights='DEFAULT')
        self.mbnet_2 = models.mobilenet_v3_small(weights='DEFAULT')

        self.mbnet_2.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.mbnet_1.classifier[3] = nn.Identity()
        self.mbnet_2.classifier[3] = nn.Identity() #nn.Linear(1024, 256)

        # self.fc_1 = nn.Linear(1280, 512) # Change the input size to reflect the modified output size
        self.fc_1 = nn.Linear(2048, 512)

        self.fc1_steer = nn.Linear(512, 128)
        self.fc1_speed = nn.Linear(512, 128)

        self.fc2_steer = nn.Linear(128, 1)
        self.fc2_speed = nn.Linear(128, 1)

        # self.fc3_steer = nn.Linear(32, 1)
        # self.fc3_speed = nn.Linear(32, 1)

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

        # speed_predict = self.fc3_speed(speed_predict)
        # steer_predict = self.fc3_steer(steer_predict)

        return speed_predict, steer_predict

def get_mobilenet_1():
    # Create an instance of the model and move it to the GPU if available
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    # Output the summary of the model
    #summary(model, [(3, 240, 424), (1, 480, 848)])

    return model

