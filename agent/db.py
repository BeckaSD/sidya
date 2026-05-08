from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import date
import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine       = create_engine(DATABASE_URL)
Session      = sessionmaker(bind=engine)
Base         = declarative_base()

class Entreprise(Base):
    __tablename__ = "entreprises"
    id      = Column(Integer, primary_key=True)
    nom     = Column(String, unique=True, nullable=False)
    voyages = relationship("Voyage", back_populates="entreprise")

class Voyage(Base):
    __tablename__ = "voyages"
    id                    = Column(Integer, primary_key=True)
    date                  = Column(Date, default=date.today)
    num_camion            = Column(String, nullable=False)
    tonnage               = Column(Float, nullable=False)
    entreprise_id         = Column(Integer, ForeignKey("entreprises.id"))
    prix_camion_par_tonne = Column(Float, nullable=False)
    prix_client_par_tonne = Column(Float, nullable=False)
    entreprise            = relationship("Entreprise", back_populates="voyages")

Base.metadata.create_all(engine)
print("✅ Base de donnees prete")
