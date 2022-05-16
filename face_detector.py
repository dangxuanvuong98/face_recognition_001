from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceDetector:
    def __init__(self) -> None:
        self.mtcnn = MTCNN()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
    def extract(self, img):
        img_cropped = self.mtcnn(img)
        if img_cropped != None:
            feature = self.resnet(img_cropped.unsqueeze(0))
        else:
            feature = None
        boxes, _ = self.mtcnn.detect(img)
        return boxes, feature