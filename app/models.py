from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Float,
    UniqueConstraint
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB

from app.database import Base


# ==================================================
# CLASS
# ==================================================

class Class(Base):

    __tablename__ = "classes"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    class_name = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    enrollments = relationship(
        "StudentEnrollment",
        back_populates="classroom",
        cascade="all, delete-orphan"
    )

    attendance_sessions = relationship(
        "AttendanceSession",
        back_populates="classroom",
        cascade="all, delete-orphan"
    )


# ==================================================
# STUDENT
# ==================================================

class Student(Base):

    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True
    )

    name = Column(
        String(150),
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    embeddings = relationship(
        "FaceEmbedding",
        back_populates="student",
        cascade="all, delete-orphan"
    )

    attendance_records = relationship(
        "AttendanceRecord",
        back_populates="student",
        cascade="all, delete-orphan"
    )

    enrollments = relationship(
        "StudentEnrollment",
        back_populates="student",
        cascade="all, delete-orphan"
    )


# ==================================================
# STUDENT ENROLLMENT
# ==================================================

class StudentEnrollment(Base):

    __tablename__ = "student_enrollments"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    class_id = Column(
        Integer,
        ForeignKey(
            "classes.id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey(
            "students.id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True
    )

    enrolled_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    classroom = relationship(
        "Class",
        back_populates="enrollments"
    )

    student = relationship(
        "Student",
        back_populates="enrollments"
    )

    __table_args__ = (
        UniqueConstraint(
            "class_id",
            "student_id",
            name="unique_student_per_class"
        ),
    )


# ==================================================
# FACE EMBEDDING
# ==================================================

class FaceEmbedding(Base):

    __tablename__ = "face_embeddings"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey(
            "students.id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True
    )

    embedding = Column(
        JSONB,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    student = relationship(
        "Student",
        back_populates="embeddings"
    )


# ==================================================
# ATTENDANCE SESSION
# ==================================================

class AttendanceSession(Base):

    __tablename__ = "attendance_sessions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # NEW: Link attendance session to class
    class_id = Column(
        Integer,
        ForeignKey(
            "classes.id",
            ondelete="CASCADE"
        ),
        nullable=True,
        index=True
    )

    # Keep temporarily for compatibility with your
    # existing session and APIs
    class_name = Column(
        String(100),
        nullable=False,
        index=True
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    classroom = relationship(
        "Class",
        back_populates="attendance_sessions"
    )

    records = relationship(
        "AttendanceRecord",
        back_populates="session",
        cascade="all, delete-orphan"
    )


# ==================================================
# ATTENDANCE RECORD
# ==================================================

class AttendanceRecord(Base):

    __tablename__ = "attendance_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    session_id = Column(
        Integer,
        ForeignKey(
            "attendance_sessions.id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey(
            "students.id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True
    )

    status = Column(
        String(20),
        nullable=False
    )

    similarity = Column(
        Float,
        nullable=True
    )

    confidence_level = Column(
        String(20),
        nullable=True
    )

    marked_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    session = relationship(
        "AttendanceSession",
        back_populates="records"
    )

    student = relationship(
        "Student",
        back_populates="attendance_records"
    )

    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "student_id",
            name="unique_student_per_session"
        ),
    )