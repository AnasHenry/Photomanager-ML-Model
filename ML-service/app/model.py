from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def generate_embedding(image):
    face = mtcnn(image)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face)

    return embedding[0].cpu().numpy().tolist()
