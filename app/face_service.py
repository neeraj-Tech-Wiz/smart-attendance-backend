from insightface.app import FaceAnalysis
import numpy as np


class FaceService:

    def __init__(self):
        self.app = None

    def load_model(self):
        """
        Load the AI model only when it is actually needed.
        """

        if self.app is None:

            print("Loading AI face recognition model...")

            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"]
            )

            self.app.prepare(
                ctx_id=-1,
                det_size=(640, 640)
            )

            print("Face recognition model loaded successfully!")

    def detect_faces(self, image):

        self.load_model()

        faces = self.app.get(image)

        results = []

        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "confidence": float(face.det_score)
            })

        return results

    def get_faces_with_embeddings(self, image):

        self.load_model()

        faces = self.app.get(image)

        results = []

        for face in faces:
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "confidence": float(face.det_score),
                "embedding": face.embedding
            })

        return results

    def get_single_face_embedding(self, image):
        """
        Used during student registration.
        Registration image must contain exactly one face.
        """

        self.load_model()

        faces = self.app.get(image)

        if len(faces) == 0:
            return None, "No face detected"

        if len(faces) > 1:
            return None, (
                "Multiple faces detected. "
                "Upload a photo with only one student."
            )

        return faces[0].embedding, None

    def compare_embeddings(self, embedding1, embedding2):
        """
        Cosine similarity between two face embeddings.
        """

        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        denominator = (
            np.linalg.norm(embedding1)
            * np.linalg.norm(embedding2)
        )

        if denominator == 0:
            return 0.0

        similarity = (
            np.dot(embedding1, embedding2)
            / denominator
        )

        return float(similarity)


# Create service object, but DON'T load AI model yet
face_service = FaceService()