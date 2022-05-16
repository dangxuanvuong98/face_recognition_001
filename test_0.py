from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_face
from facenet_pytorch.models.mtcnn import fixed_image_standardization

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN()

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()


image_path = "./test_2.jpg"

img = Image.open(image_path)

# Get cropped and prewhitened image tensor
batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
batch_boxes, batch_probs, batch_points = mtcnn.select_boxes(
    batch_boxes, batch_probs, batch_points, img, method=mtcnn.selection_method
)
img_cropped = extract_face(
    img, batch_boxes[0], mtcnn.image_size, mtcnn.margin, './output_3.jpg')
img_cropped = fixed_image_standardization(img_cropped)

print(batch_points[0])

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))

# print(img_embedding)
