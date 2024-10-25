import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image


class CNNFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        
        if pretrained:

            # Here we just use AlexNet as an example, it is pretrained on ImageNet
            # and its performance is fairly good 
            from torchvision.models import AlexNet
            self.model = AlexNet(pretrained=True)
            
            # Get the features and avgpool layer
            self.features = self.model.features
            # avgpool is used to get the average value of the features
            self.avgpool = self.model.avgpool
            # Remove the last layer, because we only need the features
            self.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        else:
            # If not pretrained, we can train it on our own dataset
            # We can also use other models like VGG, ResNet, etc. (ResNet 50 is a good choice)
            # Here we shadow the AlexNet model
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                #  - Rectified Linear Unit, it is a type of activation function (the one used by AlexNet)
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
  
                nn.Conv2d(96, 256, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(256, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(384, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                # above we have 5 convolutional layers, and 3 maxpooling layers
                # and we can use this as the input of the classifier
            )

            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

            # Here we use 4096 to match the original AlexNet model
            # but in reality it will be awfully slow to train
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1000),
            )

    #  Forward pass, just to send the input through the network
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def warp_region(image, region):
    #  Need to 'warp' (it's just resize) the region to 227 x 227

    #  Get the region
    x, y, w, h = map(int, region) # x, y, w, h are float, we need to convert them to int
    region = image[y:y+h, x:x+w]
    region_warped = cv2.resize(region, (227, 227))

    #  Convert BGR to RGB (if we use cv2 like always)
    region_warped = cv2.cvtColor(region_warped, cv2.COLOR_BGR2RGB)

    return region_warped

def preprocess_image(image):
    # Here is for ImageNet

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(image)


# Now it's the proper RCNN "architecture"
class RCNNFeatureExtractor:
    def __init__(self, use_gpu=True):
        # just to check if the GPU is available, especially for Apple Silicon (mine in this case..)
        # cuba is not available on Google Colab, just like detectron2
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = CNNFeatureExtractor(pretrained=True).to(self.device)
        self.model.eval()

    def extract_features(self, image, region):
        """
        Extract features for multiple regions in an image
        
        Args:
            image: numpy array (H, W, C) in BGR format
            regions: list of [x, y, w, h] coordinates
        
        Returns:
            features: numpy array (num_regions, 4096)
        """
        features = []

        with torch.no_grad():
            for region in region:
                warped = warp_region(image, region)

                # Convert to PIL Image (needed for torchvision)
                pil_image = Image.fromarray(warped)

                # Preprocess the image
                x = preprocess_image(pil_image)
                # Add batch dimension
                x = x.unsqueeze(0)
                # Send to device
                x = x.to(self.device)

                # Get the features
                feature = self.model(x)

                # Move to CPU and convert to numpy
                feature = feature.cpu().numpy()
                features.append(feature)
        
        # Stack all the features
        return np.vstack(features)

def demo():
    # please change the path to your own image
    image = cv2.imread("xxx")

    # Here we just use some random/dummy regions,
    # in reality we need to use a region proposal method like selective search
    region = [
        [10, 10, 100, 100],
        [50, 50, 150, 150],
        [100, 100, 200, 200],
    ]

    extractor = RCNNFeatureExtractor(use_gpu=True)

    features = extractor.extract_features(image, region)
    
    # print(f"Ectracted features: {features.shape}")

    return features

if __name__ == "__main__":
    # For the visualization if needed
    import matplotlib.pyplot as plt

    try:
        # Can even create a dummy image here
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        cv2.imwrite("dummy.jpg", dummy_image)

        features = demo()

        # Visualize first feature vector
        plt.figure(figsize=(15, 5))
        plt.plot(features[0])
        plt.title('Feature Vector Visualization (First Region)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Feature Value')
        plt.show()
    
    except Exception as e:
        print(f"Error: {e}")

                

    
        
    
            