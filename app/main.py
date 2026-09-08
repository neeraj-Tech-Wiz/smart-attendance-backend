from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import base64
import numpy as np
import cv2

from app.face_service import face_service

from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, SessionLocal, Base
from app import models

from fastapi.responses import FileResponse
from pathlib import Path


# Create database tables
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Smart Attendance AI Service"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Base64RecognitionRequest(BaseModel):
    image_base64: str
    class_id: int | None = None     

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_FILE = (
    BASE_DIR / "frontend" / "smart_attendance_frontend_index.html"
)


def read_image(image_bytes):
    np_image = np.frombuffer(image_bytes, np.uint8)

    return cv2.imdecode(
        np_image,
        cv2.IMREAD_COLOR
    )


@app.get("/" , include_in_schema=False)
def home():
    return FileResponse(FRONTEND_FILE)


# ------------------------------------------------
# GET ALL REGISTERED STUDENTS
# ------------------------------------------------

@app.get("/students")
def get_students():

    db = SessionLocal()

    try:
        students = db.query(models.Student).all()

        return {
            "total_students": len(students),
            "students": [
                {
                    "student_id": student.student_id,
                    "name": student.name,
                    "total_embeddings": len(student.embeddings)
                }
                for student in students
            ]
        }

    finally:
        db.close()


# ------------------------------------------------
# REGISTER STUDENT USING 3 FACE PHOTOS
# + AUTOMATICALLY ENROLL IN SELECTED CLASS
# ------------------------------------------------

@app.post("/register-face")
async def register_face(

    student_id: str = Form(...),

    name: str = Form(...),

    class_id: int = Form(...),

    file1: UploadFile = File(...),

    file2: UploadFile = File(...),

    file3: UploadFile = File(...)

):

    db = SessionLocal()

    try:

        # ------------------------------------------
        # Check whether student ID already exists
        # ------------------------------------------

        existing_student = (
            db.query(models.Student)
            .filter(
                models.Student.student_id == student_id
            )
            .first()
        )

        if existing_student:

            return {
                "success": False,
                "message": (
                    f"Student with ID {student_id} "
                    "already exists"
                )
            }

        # ------------------------------------------
        # Check whether selected class exists
        # ------------------------------------------

        classroom = (
            db.query(models.Class)
            .filter(
                models.Class.id == class_id
            )
            .first()
        )

        if classroom is None:

            return {
                "success": False,
                "message": "Selected class not found"
            }

        # ------------------------------------------
        # Process all 3 face photos
        # ------------------------------------------

        files = [file1, file2, file3]

        embeddings = []

        errors = []

        for file in files:

            image_bytes = await file.read()

            image = read_image(image_bytes)

            if image is None:

                errors.append(
                    f"{file.filename}: Invalid image"
                )

                continue

            embedding, error = (
                face_service.get_single_face_embedding(
                    image
                )
            )

            if error:

                errors.append(
                    f"{file.filename}: {error}"
                )

                continue

            embeddings.append(
                embedding.tolist()
            )

        # ------------------------------------------
        # Check valid photos
        # ------------------------------------------

        if len(embeddings) == 0:

            return {
                "success": False,
                "message": (
                    "No valid face photos were registered"
                ),
                "errors": errors
            }

        # ------------------------------------------
        # Create Student
        # ------------------------------------------

        new_student = models.Student(

            student_id=student_id,

            name=name

        )

        db.add(new_student)

        # Generate database ID
        db.flush()

        # ------------------------------------------
        # Save Face Embeddings
        # ------------------------------------------

        for embedding in embeddings:

            face_embedding = models.FaceEmbedding(

                student_id=new_student.id,

                embedding=embedding

            )

            db.add(face_embedding)

        # ------------------------------------------
        # Automatically Enroll Student in Class
        # ------------------------------------------

        enrollment = models.StudentEnrollment(

            class_id=classroom.id,

            student_id=new_student.id

        )

        db.add(enrollment)

        # ------------------------------------------
        # Commit everything together
        # ------------------------------------------

        db.commit()

        return {

            "success": True,

            "message": (
                f"{name} registered and enrolled successfully"
            ),

            "student": {

                "student_id": student_id,

                "name": name

            },

            "class": {

                "class_id": classroom.id,

                "class_name": classroom.class_name

            },

            "valid_photos": len(embeddings),

            "failed_photos": len(errors),

            "errors": errors

        }

    except Exception as e:

        db.rollback()

        return {

            "success": False,

            "message": "Student registration failed",

            "error": str(e)

        }

    finally:

        db.close()
# ------------------------------------------------
# RECOGNIZE MULTIPLE FACES IN CLASSROOM PHOTO
# ------------------------------------------------

@app.post("/recognize-faces")
async def recognize_faces(
    file: UploadFile = File(...)
):

    db = SessionLocal()

    try:

        print("\n========================================")
        print("FACE RECOGNITION REQUEST RECEIVED")
        print("========================================")
        print("Filename:", file.filename)
        print("Content Type:", file.content_type)

        # ------------------------------------------
        # Read uploaded image
        # ------------------------------------------

        image_bytes = await file.read()

        print("Image bytes received:", len(image_bytes))

        if not image_bytes:

            return {
                "success": False,
                "message": "Empty image received"
            }

        # ------------------------------------------
        # Convert bytes to OpenCV image
        # ------------------------------------------

        image = read_image(image_bytes)

        if image is None:

            print("ERROR: Invalid image")

            return {
                "success": False,
                "message": "Invalid image"
            }

        print("Image decoded successfully")
        print("Image shape:", image.shape)

        # ------------------------------------------
        # Detect faces + generate embeddings
        # ------------------------------------------

        detected_faces = (
            face_service.get_faces_with_embeddings(
                image
            )
        )

        print(
            "Total faces detected:",
            len(detected_faces)
        )

        # ------------------------------------------
        # Load all students from PostgreSQL
        # ------------------------------------------

        students = (
            db.query(models.Student)
            .all()
        )

        print(
            "Registered students:",
            len(students)
        )

        if len(students) == 0:

            return {

                "success": False,

                "message":
                    "No registered students found"

            }

        # ------------------------------------------
        # Confidence thresholds
        # ------------------------------------------

        HIGH_THRESHOLD = 0.60

        MEDIUM_THRESHOLD = 0.45

        # ------------------------------------------
        # Find best candidate for every detected face
        # ------------------------------------------

        face_candidates = []

        for face_index, face in enumerate(
            detected_faces
        ):

            face_embedding = face["embedding"]

            best_match = None

            best_similarity = -1

            # Compare against every student
            for student in students:

                student_best_similarity = -1

                # Compare against every registered
                # embedding for this student
                for stored_face in student.embeddings:

                    similarity = (
                        face_service.compare_embeddings(

                            face_embedding,

                            stored_face.embedding

                        )
                    )

                    student_best_similarity = max(

                        student_best_similarity,

                        similarity

                    )

                # Find best student match
                if (
                    student_best_similarity
                    > best_similarity
                ):

                    best_similarity = (
                        student_best_similarity
                    )

                    best_match = {

                        "student_id":
                            student.student_id,

                        "name":
                            student.name

                    }

            print(
                f"Face {face_index} | "
                f"Best Match: "
                f"{best_match['name'] if best_match else 'None'} | "
                f"Similarity: {best_similarity}"
            )

            face_candidates.append({

                "face_index":
                    face_index,

                "bbox":
                    face["bbox"],

                "detection_confidence":
                    face["confidence"],

                "best_match":
                    best_match,

                "similarity":
                    best_similarity

            })

        # ------------------------------------------
        # Prevent duplicate recognition
        # Keep strongest face for each student
        # ------------------------------------------

        student_best_faces = {}

        for candidate in face_candidates:

            if (

                candidate["best_match"]
                is not None

                and

                candidate["similarity"]
                >= MEDIUM_THRESHOLD

            ):

                matched_student_id = (

                    candidate["best_match"]
                    ["student_id"]

                )

                # Keep only strongest recognition
                # for each student
                if (

                    matched_student_id
                    not in student_best_faces

                    or

                    candidate["similarity"]

                    >

                    student_best_faces[
                        matched_student_id
                    ]["similarity"]

                ):

                    student_best_faces[
                        matched_student_id
                    ] = candidate

        # ------------------------------------------
        # Build final results
        # ------------------------------------------

        results = []

        for candidate in face_candidates:

            similarity = candidate["similarity"]

            best_match = candidate["best_match"]

            recognized = False

            confidence_level = "UNKNOWN"

            matched_student_id = None

            name = "Unknown"

            is_best_student_match = (

                best_match is not None

                and

                best_match["student_id"]
                in student_best_faces

                and

                student_best_faces[
                    best_match["student_id"]
                ]["face_index"]

                ==

                candidate["face_index"]

            )

            # --------------------------------------
            # Recognize student
            # --------------------------------------

            if is_best_student_match:

                recognized = True

                matched_student_id = (
                    best_match["student_id"]
                )

                name = best_match["name"]

                if similarity >= HIGH_THRESHOLD:

                    confidence_level = "HIGH"

                else:

                    confidence_level = "MEDIUM"

            results.append({

                "bbox":
                    candidate["bbox"],

                "detection_confidence":
                    round(
                        candidate[
                            "detection_confidence"
                        ],
                        4
                    ),

                "recognized":
                    recognized,

                "confidence_level":
                    confidence_level,

                "student_id":
                    matched_student_id,

                "name":
                    name,

                "similarity":
                    round(similarity, 4)

                    if similarity != -1

                    else None

            })

        # ------------------------------------------
        # Get recognized students
        # ------------------------------------------

        recognized_students = [

            result

            for result in results

            if result["recognized"]

        ]

        print("\n========================================")
        print("RECOGNITION COMPLETE")
        print("========================================")
        print(
            "Total Faces:",
            len(detected_faces)
        )
        print(
            "Recognized Faces:",
            len(recognized_students)
        )
        print("========================================\n")

        # ------------------------------------------
        # Return response to Mobile App
        # ------------------------------------------

        return {

            "success": True,

            "total_faces":
                len(detected_faces),

            "recognized_faces":
                len(recognized_students),

            "high_confidence_faces":

                sum(

                    1

                    for result
                    in recognized_students

                    if result[
                        "confidence_level"
                    ] == "HIGH"

                ),

            "medium_confidence_faces":

                sum(

                    1

                    for result
                    in recognized_students

                    if result[
                        "confidence_level"
                    ] == "MEDIUM"

                ),

            "faces":
                results

        }

    except Exception as e:

        print("\n========================================")
        print("RECOGNITION ERROR")
        print(str(e))
        print("========================================\n")

        return {

            "success": False,

            "message": str(e)

        }

    finally:

        db.close()
# ==================================================
# MARK ATTENDANCE USING CLASS + BASE64 GROUP PHOTO
# ==================================================

@app.post("/mark-attendance")
async def mark_attendance(
    request: dict
):

    db = SessionLocal()

    try:

        # ------------------------------------------
        # Get request data
        # ------------------------------------------

        class_id = request.get("class_id")

        image_base64 = request.get("image_base64")

        if not class_id:

            return {
                "success": False,
                "message": "class_id is required"
            }

        if not image_base64:

            return {
                "success": False,
                "message": "image_base64 is required"
            }

        # ------------------------------------------
        # Remove Base64 prefix if present
        # Example:
        # data:image/jpeg;base64,/9j/4AAQ...
        # ------------------------------------------

        if "," in image_base64:

            image_base64 = image_base64.split(",", 1)[1]

        # ------------------------------------------
        # Decode Base64 image
        # ------------------------------------------

        try:

            image_bytes = base64.b64decode(
                image_base64
            )

        except Exception as e:

            return {

                "success": False,

                "message": "Invalid Base64 image",

                "error": str(e)

            }

        # ------------------------------------------
        # Convert bytes to OpenCV image
        # ------------------------------------------

        image = read_image(image_bytes)

        if image is None:

            return {

                "success": False,

                "message": "Invalid image"

            }

        # ------------------------------------------
        # Find the class
        # ------------------------------------------

        classroom = (

            db.query(models.Class)

            .filter(
                models.Class.id == class_id
            )

            .first()

        )

        if classroom is None:

            return {

                "success": False,

                "message": "Class not found"

            }

        # ------------------------------------------
        # Get enrolled students only
        # ------------------------------------------

        enrolled_students = [

            enrollment.student

            for enrollment
            in classroom.enrollments

        ]

        if len(enrolled_students) == 0:

            return {

                "success": False,

                "message":
                    "No students enrolled in this class"

            }

        # ------------------------------------------
        # Detect faces
        # ------------------------------------------

        detected_faces = (

            face_service
            .get_faces_with_embeddings(image)

        )

        # ------------------------------------------
        # Thresholds
        # ------------------------------------------

        HIGH_THRESHOLD = 0.60

        MEDIUM_THRESHOLD = 0.45

        # ------------------------------------------
        # Find best match for every detected face
        # Only compare enrolled students
        # ------------------------------------------

        face_candidates = []

        for face_index, face in enumerate(
            detected_faces
        ):

            face_embedding = face["embedding"]

            best_match = None

            best_similarity = -1

            # --------------------------------------
            # Compare with enrolled students
            # --------------------------------------

            for student in enrolled_students:

                student_best_similarity = -1

                # Compare against all registered
                # face embeddings of this student

                for stored_face in student.embeddings:

                    similarity = (

                        face_service
                        .compare_embeddings(

                            face_embedding,

                            stored_face.embedding

                        )

                    )

                    student_best_similarity = max(

                        student_best_similarity,

                        similarity

                    )

                # ----------------------------------
                # Keep best student
                # ----------------------------------

                if (
                    student_best_similarity
                    > best_similarity
                ):

                    best_similarity = (
                        student_best_similarity
                    )

                    best_match = {

                        "student_db_id":
                            student.id,

                        "student_id":
                            student.student_id,

                        "name":
                            student.name

                    }

            face_candidates.append({

                "face_index":
                    face_index,

                "bbox":
                    face["bbox"],

                "best_match":
                    best_match,

                "similarity":
                    best_similarity

            })

        # ------------------------------------------
        # Prevent duplicate recognition
        # Keep strongest match for each student
        # ------------------------------------------

        student_best_faces = {}

        for candidate in face_candidates:

            if (

                candidate["best_match"]
                is not None

                and

                candidate["similarity"]
                >= MEDIUM_THRESHOLD

            ):

                matched_student_id = (

                    candidate["best_match"]
                    ["student_id"]

                )

                # Keep only strongest face match

                if (

                    matched_student_id
                    not in student_best_faces

                    or

                    candidate["similarity"]

                    >

                    student_best_faces[
                        matched_student_id
                    ]["similarity"]

                ):

                    student_best_faces[
                        matched_student_id
                    ] = candidate

        # ------------------------------------------
        # Create attendance session
        # ------------------------------------------

        attendance_session = (

            models.AttendanceSession(

                class_id=classroom.id,

                class_name=classroom.class_name

            )

        )

        db.add(attendance_session)

        db.flush()

        # ------------------------------------------
        # Create attendance for EVERY enrolled student
        # ------------------------------------------

        attendance_results = []

        for student in enrolled_students:

            recognized_candidate = (

                student_best_faces.get(

                    student.student_id

                )

            )

            # ======================================
            # PRESENT
            # ======================================

            if recognized_candidate:

                similarity = (

                    recognized_candidate[
                        "similarity"
                    ]

                )

                confidence_level = (

                    "HIGH"

                    if similarity >= HIGH_THRESHOLD

                    else "MEDIUM"

                )

                attendance_record = (

                    models.AttendanceRecord(

                        session_id=
                            attendance_session.id,

                        student_id=
                            student.id,

                        status="PRESENT",

                        similarity=
                            similarity,

                        confidence_level=
                            confidence_level

                    )

                )

                attendance_results.append({

                    "student_id":
                        student.student_id,

                    "name":
                        student.name,

                    "status":
                        "PRESENT",

                    "confidence_level":
                        confidence_level,

                    "similarity":
                        round(similarity, 4)

                })

            # ======================================
            # ABSENT
            # ======================================

            else:

                attendance_record = (

                    models.AttendanceRecord(

                        session_id=
                            attendance_session.id,

                        student_id=
                            student.id,

                        status="ABSENT",

                        similarity=None,

                        confidence_level=None

                    )

                )

                attendance_results.append({

                    "student_id":
                        student.student_id,

                    "name":
                        student.name,

                    "status":
                        "ABSENT",

                    "confidence_level":
                        None,

                    "similarity":
                        None

                })

            db.add(attendance_record)

        # ------------------------------------------
        # Save everything
        # ------------------------------------------

        db.commit()

        # ------------------------------------------
        # Statistics
        # ------------------------------------------

        present_count = sum(

            1

            for result in attendance_results

            if result["status"] == "PRESENT"

        )

        absent_count = sum(

            1

            for result in attendance_results

            if result["status"] == "ABSENT"

        )

        # ------------------------------------------
        # Final response
        # ------------------------------------------

        return {

            "success": True,

            "message":
                "Attendance marked successfully",

            "session_id":
                attendance_session.id,

            "class": {

                "class_id":
                    classroom.id,

                "class_name":
                    classroom.class_name

            },

            "total_faces_detected":
                len(detected_faces),

            "total_students":
                len(enrolled_students),

            "students_present":
                present_count,

            "students_absent":
                absent_count,

            "attendance":
                attendance_results

        }

    # ----------------------------------------------
    # Error handling
    # ----------------------------------------------

    except Exception as e:

        db.rollback()

        print(
            "MARK ATTENDANCE ERROR:",
            str(e)
        )

        return {

            "success": False,

            "message":
                "Failed to mark attendance",

            "error":
                str(e)

        }

    finally:

        db.close()
# ==================================================
# GET ALL ATTENDANCE SESSIONS
# ==================================================

@app.get("/attendance/sessions")
def get_attendance_sessions():

    db = SessionLocal()

    try:

        sessions = (
            db.query(models.AttendanceSession)
            .order_by(
                models.AttendanceSession.created_at.desc()
            )
            .all()
        )

        return {
            "total_sessions": len(sessions),
            "sessions": [
                {
                    "session_id": session.id,
                    "class_name": session.class_name,
                    "created_at": session.created_at,
                    "total_present": len(session.records)
                }
                for session in sessions
            ]
        }

    finally:

        db.close() 

# ==================================================
# GET ATTENDANCE DETAILS FOR ONE SESSION
# ==================================================

@app.get("/attendance/session/{session_id}")
def get_attendance_session(session_id: int):

    db = SessionLocal()

    try:

        session = (
            db.query(models.AttendanceSession)
            .filter(
                models.AttendanceSession.id == session_id
            )
            .first()
        )

        if session is None:

            return {
                "success": False,
                "message": "Attendance session not found"
            }

        attendance = []

        for record in session.records:

            attendance.append({
                "student_id": record.student.student_id,
                "name": record.student.name,
                "status": record.status,
                "similarity": record.similarity,
                "confidence_level": record.confidence_level,
                "marked_at": record.marked_at
            })

        return {
            "success": True,

            "session": {
                "session_id": session.id,
                "class_name": session.class_name,
                "created_at": session.created_at,
                "students_present": len(attendance)
            },

            "attendance": attendance
        }

    finally:

        db.close()

# ==================================================
# GET STUDENT ATTENDANCE HISTORY
# ==================================================

@app.get("/attendance/student/{student_id}")
def get_student_attendance(student_id: str):

    db = SessionLocal()

    try:

        # Find student
        student = (
            db.query(models.Student)
            .filter(
                models.Student.student_id == student_id
            )
            .first()
        )

        if student is None:
            return {
                "success": False,
                "message": f"Student {student_id} not found"
            }

        # Get all attendance records for this student
        records = (
            db.query(models.AttendanceRecord)
            .filter(
                models.AttendanceRecord.student_id == student.id
            )
            .order_by(
                models.AttendanceRecord.marked_at.desc()
            )
            .all()
        )

        total_present = sum(
            1 for record in records
            if record.status == "PRESENT"
        )

        total_absent = sum(
            1 for record in records
            if record.status == "ABSENT"
        )

        total_records = len(records)

        # Calculate percentage
        attendance_percentage = 0

        if total_records > 0:
            attendance_percentage = round(
                (total_present / total_records) * 100,
                2
            )

        # Build history
        history = []

        for record in records:

            history.append({
                "session_id": record.session.id,
                "class_name": record.session.class_name,
                "status": record.status,
                "similarity": round(record.similarity, 4)
                if record.similarity is not None else None,
                "confidence_level": record.confidence_level,
                "marked_at": record.marked_at
            })

        return {
            "success": True,

            "student": {
                "student_id": student.student_id,
                "name": student.name
            },

            "statistics": {
                "total_records": total_records,
                "present": total_present,
                "absent": total_absent,
                "attendance_percentage": attendance_percentage
            },

            "history": history
        }

    finally:

        db.close()

# ==================================================
# CREATE CLASS
# ==================================================

@app.post("/classes")
def create_class(class_name: str = Form(...)):

    db = SessionLocal()

    try:

        # Remove accidental spaces
        class_name = class_name.strip()

        if not class_name:
            return {
                "success": False,
                "message": "Class name cannot be empty"
            }

        existing_class = (
            db.query(models.Class)
            .filter(models.Class.class_name == class_name)
            .first()
        )

        if existing_class:
            return {
                "success": False,
                "message": "Class already exists"
            }

        new_class = models.Class(
            class_name=class_name
        )

        db.add(new_class)
        db.commit()
        db.refresh(new_class)

        return {
            "success": True,
            "message": "Class created successfully",
            "class": {
                "class_id": new_class.id,
                "class_name": new_class.class_name,
                "created_at": new_class.created_at
            }
        }

    finally:
        db.close()

# ==================================================
# GET ALL CLASSES
# ==================================================

@app.get("/classes")
def get_classes():

    db = SessionLocal()

    try:

        classes = (
            db.query(models.Class)
            .order_by(models.Class.created_at.desc())
            .all()
        )

        return {
            "total_classes": len(classes),
            "classes": [
                {
                    "class_id": classroom.id,
                    "class_name": classroom.class_name,
                    "created_at": classroom.created_at,
                    "total_students": len(classroom.enrollments)
                }
                for classroom in classes
            ]
        }

    finally:
        db.close()

# ==================================================
# ENROLL STUDENT IN CLASS
# ==================================================

@app.post("/classes/{class_id}/enroll/{student_id}")
def enroll_student(
    class_id: int,
    student_id: str
):

    db = SessionLocal()

    try:

        classroom = (
            db.query(models.Class)
            .filter(models.Class.id == class_id)
            .first()
        )

        if classroom is None:
            return {
                "success": False,
                "message": "Class not found"
            }

        student = (
            db.query(models.Student)
            .filter(models.Student.student_id == student_id)
            .first()
        )

        if student is None:
            return {
                "success": False,
                "message": "Student not found"
            }

        existing_enrollment = (
            db.query(models.StudentEnrollment)
            .filter(
                models.StudentEnrollment.class_id == class_id,
                models.StudentEnrollment.student_id == student.id
            )
            .first()
        )

        if existing_enrollment:
            return {
                "success": False,
                "message": "Student is already enrolled in this class"
            }

        enrollment = models.StudentEnrollment(
            class_id=classroom.id,
            student_id=student.id
        )

        db.add(enrollment)
        db.commit()

        return {
            "success": True,
            "message": "Student enrolled successfully",
            "class_name": classroom.class_name,
            "student": {
                "student_id": student.student_id,
                "name": student.name
            }
        }

    finally:
        db.close()

# ==================================================
# GET CLASS STUDENTS
# ==================================================

@app.get("/classes/{class_id}/students")
def get_class_students(class_id: int):

    db = SessionLocal()

    try:

        classroom = (
            db.query(models.Class)
            .filter(models.Class.id == class_id)
            .first()
        )

        if classroom is None:
            return {
                "success": False,
                "message": "Class not found"
            }

        students = []

        for enrollment in classroom.enrollments:

            student = enrollment.student

            students.append({
                "student_id": student.student_id,
                "name": student.name,
                "enrolled_at": enrollment.enrolled_at
            })

        return {
            "success": True,
            "class": {
                "class_id": classroom.id,
                "class_name": classroom.class_name
            },
            "total_students": len(students),
            "students": students
        }

    finally:
        db.close()

# ------------------------------------------------
# RECOGNIZE FACES USING BASE64 IMAGE
# ------------------------------------------------

@app.post("/recognize-faces-base64")
async def recognize_faces_base64(request: dict):

    try:
        import base64
        import numpy as np

        # Get Base64 image
        image_base64 = request.get("image_base64")

        if not image_base64:
            return {
                "success": False,
                "message": "Image is required"
            }

        # Remove data URI prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode Base64
        image_bytes = base64.b64decode(image_base64)

        # Convert bytes to image
        image = read_image(image_bytes)

        if image is None:
            return {
                "success": False,
                "message": "Invalid image"
            }

        # ------------------------------------------
        # Detect faces
        # ------------------------------------------

        detected_faces = (
            face_service.get_faces_with_embeddings(image)
        )

        # ------------------------------------------
        # Load registered students
        # ------------------------------------------

        db = SessionLocal()

        try:

            students = db.query(
                models.Student
            ).all()

            if len(students) == 0:
                return {
                    "success": False,
                    "message": "No registered students found"
                }

            HIGH_THRESHOLD = 0.60
            MEDIUM_THRESHOLD = 0.45

            face_candidates = []

            # --------------------------------------
            # Compare every detected face
            # --------------------------------------

            for face_index, face in enumerate(detected_faces):

                face_embedding = face["embedding"]

                best_match = None
                best_similarity = -1

                for student in students:

                    student_best_similarity = -1

                    for stored_face in student.embeddings:

                        similarity = (
                            face_service.compare_embeddings(
                                face_embedding,
                                stored_face.embedding
                            )
                        )

                        student_best_similarity = max(
                            student_best_similarity,
                            similarity
                        )

                    if student_best_similarity > best_similarity:

                        best_similarity = student_best_similarity

                        best_match = {
                            "student_id": student.student_id,
                            "name": student.name
                        }

                face_candidates.append({
                    "face_index": face_index,
                    "bbox": face["bbox"],
                    "detection_confidence": face["confidence"],
                    "best_match": best_match,
                    "similarity": best_similarity
                })

            # --------------------------------------
            # Remove duplicate student recognition
            # --------------------------------------

            student_best_faces = {}

            for candidate in face_candidates:

                if (
                    candidate["best_match"] is not None
                    and candidate["similarity"] >= MEDIUM_THRESHOLD
                ):

                    student_id = candidate["best_match"]["student_id"]

                    if (
                        student_id not in student_best_faces
                        or candidate["similarity"]
                        > student_best_faces[student_id]["similarity"]
                    ):

                        student_best_faces[student_id] = candidate

            # --------------------------------------
            # Build final recognition results
            # --------------------------------------

            results = []

            for candidate in face_candidates:

                similarity = candidate["similarity"]
                best_match = candidate["best_match"]

                recognized = False
                confidence_level = "UNKNOWN"
                student_id = None
                name = "Unknown"

                is_best_match = (
                    best_match is not None
                    and best_match["student_id"]
                    in student_best_faces
                    and student_best_faces[
                        best_match["student_id"]
                    ]["face_index"]
                    == candidate["face_index"]
                )

                if is_best_match:

                    recognized = True
                    student_id = best_match["student_id"]
                    name = best_match["name"]

                    if similarity >= HIGH_THRESHOLD:
                        confidence_level = "HIGH"
                    else:
                        confidence_level = "MEDIUM"

                results.append({
                    "bbox": candidate["bbox"],
                    "detection_confidence": round(
                        candidate["detection_confidence"], 4
                    ),
                    "recognized": recognized,
                    "confidence_level": confidence_level,
                    "student_id": student_id,
                    "name": name,
                    "similarity": (
                        round(similarity, 4)
                        if similarity != -1
                        else None
                    )
                })

            recognized_students = [
                result
                for result in results
                if result["recognized"]
            ]

            return {
                "success": True,
                "total_faces": len(detected_faces),
                "recognized_faces": len(recognized_students),
                "high_confidence_faces": sum(
                    1
                    for result in recognized_students
                    if result["confidence_level"] == "HIGH"
                ),
                "medium_confidence_faces": sum(
                    1
                    for result in recognized_students
                    if result["confidence_level"] == "MEDIUM"
                ),
                "faces": results
            }

        finally:
            db.close()

    except Exception as e:

        print("RECOGNITION ERROR:", str(e))

        return {
            "success": False,
            "message": str(e)
        }

