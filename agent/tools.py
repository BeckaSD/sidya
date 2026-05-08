from langchain_core.tools import tool
from sqlalchemy import extract
from db import Session, Entreprise, Voyage
from pdf_gen import generate_report_pdf, generate_summary_pdf
from datetime import date


def _find_entreprise(db, nom: str):
    return db.query(Entreprise).filter(Entreprise.nom.ilike(nom)).first()

def _all_entreprises(db):
    return [e.nom for e in db.query(Entreprise).order_by(Entreprise.nom).all()]

def _voyages_to_data(voyages):
    return [{
        'id':          v.id,
        'date':        str(v.date),
        'num_camion':  v.num_camion,
        'tonnage':     v.tonnage,
        'prix_camion': v.prix_camion_par_tonne,
        'prix_client': v.prix_client_par_tonne,
    } for v in voyages]


@tool
def list_entreprises() -> str:
    """Retourne la liste de toutes les entreprises enregistrees."""
    db = Session()
    try:
        noms = _all_entreprises(db)
        if not noms:
            return "❌ Aucune entreprise enregistree."
        return "📋 Entreprises disponibles:\n" + "\n".join(f"  • {n}" for n in noms)
    finally:
        db.close()


@tool
def save_trip(num_camion: str, tonnage: float, entreprise: str,
              prix_camion: float, prix_client: float,
              unite: str = "tonne") -> str:
    """Enregistre un voyage de camion dans la base de donnees.
    unite = 'kg' si tonnage en kilogrammes, 'tonne' si deja en tonnes."""
    db = Session()
    try:
        if unite.lower() in ("kg", "kilogramme", "kilogrammes"):
            tonnage_tonnes = tonnage / 1000
            unite_log = f"{tonnage} kg → {tonnage_tonnes}t"
        else:
            tonnage_tonnes = tonnage
            unite_log = f"{tonnage_tonnes}t"

        ent = db.query(Entreprise).filter_by(nom=entreprise).first()
        if not ent:
            ent = Entreprise(nom=entreprise)
            db.add(ent)
            db.flush()

        voyage = Voyage(
            date=date.today(),
            num_camion=num_camion,
            tonnage=tonnage_tonnes,
            entreprise_id=ent.id,
            prix_camion_par_tonne=prix_camion,
            prix_client_par_tonne=prix_client,
        )
        db.add(voyage)
        db.commit()
        return (f"✅ Voyage enregistre:\n"
                f"   Camion     : {num_camion}\n"
                f"   Tonnage    : {unite_log}\n"
                f"   Entreprise : {entreprise}\n"
                f"   Prix camion: {prix_camion} MRU/t\n"
                f"   Prix client: {prix_client} MRU/t\n"
                f"   Date       : {date.today()}")
    finally:
        db.close()


@tool
def delete_trip(voyage_id: int) -> str:
    """Supprime un voyage de la base de donnees par son ID."""
    db = Session()
    try:
        voyage = db.query(Voyage).filter_by(id=voyage_id).first()
        if not voyage:
            return f"❌ Aucun voyage trouve avec l ID {voyage_id}."
        ent  = db.query(Entreprise).filter_by(id=voyage.entreprise_id).first()
        info = (f"   ID         : {voyage.id}\n"
                f"   Date       : {voyage.date}\n"
                f"   Camion     : {voyage.num_camion}\n"
                f"   Tonnage    : {voyage.tonnage}t\n"
                f"   Entreprise : {ent.nom if ent else '?'}")
        db.delete(voyage)
        db.commit()
        return f"✅ Voyage supprime:\n{info}"
    finally:
        db.close()


@tool
def set_price(num_camion: str, prix_camion: float, prix_client: float) -> str:
    """Met a jour les prix d'un voyage existant."""
    db = Session()
    try:
        voyage = (db.query(Voyage)
                  .filter_by(num_camion=num_camion)
                  .order_by(Voyage.date.desc())
                  .first())
        if not voyage:
            return f"❌ Aucun voyage trouve pour camion {num_camion}"
        voyage.prix_camion_par_tonne = prix_camion
        voyage.prix_client_par_tonne = prix_client
        db.commit()
        return (f"✅ Prix mis a jour — Camion {num_camion}:\n"
                f"   Prix camion: {prix_camion} MRU/t\n"
                f"   Prix client: {prix_client} MRU/t")
    finally:
        db.close()


@tool
def get_report(entreprise: str, mois: int, annee: int) -> str:
    """Retourne le rapport mensuel d'une entreprise avec IDs.
    Si nom exact introuvable, retourne la liste des entreprises disponibles."""
    db = Session()
    try:
        ent = _find_entreprise(db, entreprise)
        if not ent:
            noms = _all_entreprises(db)
            return (f"❌ Entreprise '{entreprise}' introuvable.\n"
                    f"📋 Entreprises disponibles:\n" +
                    "\n".join(f"  • {n}" for n in noms) +
                    "\n\nChoisir la plus proche et rappeler get_report avec le nom exact.")

        voyages = (db.query(Voyage)
                   .filter_by(entreprise_id=ent.id)
                   .filter(extract('month', Voyage.date) == mois)
                   .filter(extract('year',  Voyage.date) == annee)
                   .order_by(Voyage.date).all())
        if not voyages:
            return f"❌ Aucun voyage pour {ent.nom} en {mois}/{annee}."

        total_tonnes  = sum(v.tonnage for v in voyages)
        total_montant = sum(v.tonnage * v.prix_client_par_tonne for v in voyages)
        lignes = [f"📋 RAPPORT {ent.nom.upper()} — {mois}/{annee}",
                  "─────────────────────────────"]
        for v in voyages:
            montant = v.tonnage * v.prix_client_par_tonne
            lignes.append(f"🆔 {v.id} | 📅 {v.date} | 🚛 {v.num_camion} | {v.tonnage}t | {montant:.0f} MRU")
        lignes.append("─────────────────────────────")
        lignes.append(f"📦 Total tonnage : {total_tonnes}t")
        lignes.append(f"💰 Total montant : {total_montant:.0f} MRU")
        return "\n".join(lignes)
    finally:
        db.close()


@tool
def get_profit(entreprise: str, mois: int, annee: int) -> str:
    """Calcule le benefice net pour une entreprise sur un mois donne.
    Si nom exact introuvable, retourne la liste des entreprises disponibles."""
    db = Session()
    try:
        ent = _find_entreprise(db, entreprise)
        if not ent:
            noms = _all_entreprises(db)
            return (f"❌ Entreprise '{entreprise}' introuvable.\n"
                    f"📋 Entreprises disponibles:\n" +
                    "\n".join(f"  • {n}" for n in noms) +
                    "\n\nChoisir la plus proche et rappeler get_profit avec le nom exact.")

        voyages = (db.query(Voyage)
                   .filter_by(entreprise_id=ent.id)
                   .filter(extract('month', Voyage.date) == mois)
                   .filter(extract('year',  Voyage.date) == annee).all())
        if not voyages:
            return f"❌ Aucun voyage pour {ent.nom} en {mois}/{annee}."

        recettes = sum(v.tonnage * v.prix_client_par_tonne for v in voyages)
        couts    = sum(v.tonnage * v.prix_camion_par_tonne  for v in voyages)
        benefice = recettes - couts
        return (f"💹 BENEFICE {ent.nom.upper()} — {mois}/{annee}\n"
                f"─────────────────────────────\n"
                f"💰 Recettes client : {recettes:.0f} MRU\n"
                f"🚛 Couts camions   : {couts:.0f} MRU\n"
                f"─────────────────────────────\n"
                f"✅ Benefice net    : {benefice:.0f} MRU")
    finally:
        db.close()


@tool
def get_report_pdf(entreprise: str, mois: int, annee: int) -> str:
    """Genere un PDF du rapport mensuel d'une seule entreprise.
    Appelle cet outil quand l utilisateur demande le rapport PDF d'une entreprise precise."""
    db = Session()
    try:
        ent = _find_entreprise(db, entreprise)
        if not ent:
            noms = _all_entreprises(db)
            return (f"❌ Entreprise '{entreprise}' introuvable.\n"
                    f"📋 Entreprises disponibles:\n" +
                    "\n".join(f"  • {n}" for n in noms) +
                    "\n\nChoisir la plus proche et rappeler get_report_pdf avec le nom exact.")

        voyages = (db.query(Voyage)
                   .filter_by(entreprise_id=ent.id)
                   .filter(extract('month', Voyage.date) == mois)
                   .filter(extract('year',  Voyage.date) == annee)
                   .order_by(Voyage.date).all())
        if not voyages:
            return f"❌ Aucun voyage pour {ent.nom} en {mois}/{annee}."

        pdf_path = generate_report_pdf(ent.nom, mois, annee,
                                        _voyages_to_data(voyages), mode="rapport")
        return f"PDF_PATH:{pdf_path}"
    finally:
        db.close()


@tool
def get_profit_pdf(entreprise: str, mois: int, annee: int) -> str:
    """Genere un PDF du benefice mensuel d'une seule entreprise.
    Appelle cet outil quand l utilisateur demande le benefice PDF d'une entreprise precise."""
    db = Session()
    try:
        ent = _find_entreprise(db, entreprise)
        if not ent:
            noms = _all_entreprises(db)
            return (f"❌ Entreprise '{entreprise}' introuvable.\n"
                    f"📋 Entreprises disponibles:\n" +
                    "\n".join(f"  • {n}" for n in noms) +
                    "\n\nChoisir la plus proche et rappeler get_profit_pdf avec le nom exact.")

        voyages = (db.query(Voyage)
                   .filter_by(entreprise_id=ent.id)
                   .filter(extract('month', Voyage.date) == mois)
                   .filter(extract('year',  Voyage.date) == annee).all())
        if not voyages:
            return f"❌ Aucun voyage pour {ent.nom} en {mois}/{annee}."

        pdf_path = generate_report_pdf(ent.nom, mois, annee,
                                        _voyages_to_data(voyages), mode="benefice")
        return f"PDF_PATH:{pdf_path}"
    finally:
        db.close()


@tool
def get_summary_pdf(mois: int, annee: int) -> str:
    """Genere un PDF global avec toutes les entreprises — tableau recap + detail par entreprise.
    Appelle cet outil quand l utilisateur demande un resume global, toutes entreprises, ou synthese PDF."""
    db = Session()
    try:
        entreprises = db.query(Entreprise).order_by(Entreprise.nom).all()
        if not entreprises:
            return "❌ Aucune entreprise enregistree."

        entreprises_data = []
        for ent in entreprises:
            voyages = (db.query(Voyage)
                       .filter_by(entreprise_id=ent.id)
                       .filter(extract('month', Voyage.date) == mois)
                       .filter(extract('year',  Voyage.date) == annee)
                       .order_by(Voyage.date).all())
            if not voyages:
                continue

            total_tonnes  = sum(v.tonnage for v in voyages)
            total_montant = sum(v.tonnage * v.prix_client_par_tonne for v in voyages)
            total_couts   = sum(v.tonnage * v.prix_camion_par_tonne  for v in voyages)
            benefice      = total_montant - total_couts

            entreprises_data.append({
                'nom':          ent.nom,
                'nb_voyages':   len(voyages),
                'voyages':      _voyages_to_data(voyages),
                'total_tonnes': total_tonnes,
                'total_montant':total_montant,
                'total_couts':  total_couts,
                'benefice':     benefice,
            })

        if not entreprises_data:
            return f"❌ Aucun voyage enregistre en {mois}/{annee}."

        pdf_path = generate_summary_pdf(mois, annee, entreprises_data)
        return f"PDF_PATH:{pdf_path}"
    finally:
        db.close()
