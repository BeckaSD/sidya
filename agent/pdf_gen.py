from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import date
import os

OUTPUT_DIR = "/tmp/sidya_pdfs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_report_pdf(entreprise: str, mois: int, annee: int,
                         voyages: list, mode: str = "rapport") -> str:
    """PDF pour une seule entreprise — rapport ou benefice."""
    filename = f"{OUTPUT_DIR}/{mode}_{entreprise}_{mois}_{annee}.pdf"
    doc      = SimpleDocTemplate(filename, pagesize=A4,
                                  rightMargin=2*cm, leftMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
    styles   = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('title', parent=styles['Title'],
                                  fontSize=16, spaceAfter=4,
                                  textColor=colors.HexColor('#1a1a2e'))
    sub_style   = ParagraphStyle('sub', parent=styles['Normal'],
                                  fontSize=10, spaceAfter=2,
                                  textColor=colors.HexColor('#555'))

    titre = "RAPPORT MENSUEL" if mode == "rapport" else "BENEFICE MENSUEL"
    elements.append(Paragraph(titre, title_style))
    elements.append(Paragraph(f"Entreprise : {entreprise.upper()}", sub_style))
    elements.append(Paragraph(f"Periode    : {mois:02d}/{annee}", sub_style))
    elements.append(Paragraph(f"Genere le  : {date.today().strftime('%d/%m/%Y')}", sub_style))
    elements.append(Spacer(1, 0.4*cm))

    header = ['ID', 'Date', 'Camion', 'Tonnage', 'Prix/t', 'Montant']
    rows   = [header]

    total_tonnes  = 0
    total_montant = 0
    total_couts   = 0

    for v in voyages:
        montant        = v['tonnage'] * v['prix_client']
        total_tonnes  += v['tonnage']
        total_montant += montant
        total_couts   += v['tonnage'] * v['prix_camion']
        rows.append([
            str(v['id']),
            str(v['date']),
            v['num_camion'],
            f"{v['tonnage']:.2f} t",
            f"{v['prix_client']:.0f} MRU",
            f"{montant:.0f} MRU",
        ])
    rows.append(['', '', '', f"{total_tonnes:.2f} t", 'TOTAL', f"{total_montant:.0f} MRU"])

    table = Table(rows, colWidths=[1.2*cm, 2.8*cm, 3*cm, 2.5*cm, 2.8*cm, 3.7*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0),  (-1, 0),  colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',      (0, 0),  (-1, 0),  colors.white),
        ('FONTNAME',       (0, 0),  (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0),  (-1, -1), 9),
        ('ALIGN',          (0, 0),  (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1),  (-1, -2), [colors.white, colors.HexColor('#f0f4ff')]),
        ('BACKGROUND',     (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
        ('FONTNAME',       (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID',           (0, 0),  (-1, -1), 0.5, colors.HexColor('#cccccc')),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.4*cm))

    if mode == "benefice":
        benefice   = total_montant - total_couts
        bene_data  = [
            ['Recettes client', f"{total_montant:.0f} MRU"],
            ['Couts camions',   f"{total_couts:.0f} MRU"],
            ['Benefice net',    f"{benefice:.0f} MRU"],
        ]
        bene_table = Table(bene_data, colWidths=[8*cm, 6*cm])
        bene_table.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0),  (-1, -2), colors.HexColor('#f5f5f5')),
            ('BACKGROUND',    (0, -1), (-1, -1), colors.HexColor('#c8e6c9')),
            ('FONTNAME',      (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0),  (-1, -1), 11),
            ('ALIGN',         (1, 0),  (1, -1),  'RIGHT'),
            ('GRID',          (0, 0),  (-1, -1), 0.5, colors.grey),
            ('TOPPADDING',    (0, 0),  (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0),  (-1, -1), 6),
        ]))
        elements.append(bene_table)

    doc.build(elements)
    return filename


def generate_summary_pdf(mois: int, annee: int, entreprises_data: list) -> str:
    """
    PDF global toutes entreprises.
    entreprises_data = liste de dicts:
      { nom, voyages: [...], total_tonnes, total_montant, total_couts, benefice }
    """
    filename = f"{OUTPUT_DIR}/resume_global_{mois}_{annee}.pdf"
    doc      = SimpleDocTemplate(filename, pagesize=A4,
                                  rightMargin=2*cm, leftMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
    styles   = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('title', parent=styles['Title'],
                                  fontSize=18, spaceAfter=4,
                                  textColor=colors.HexColor('#1a1a2e'))
    ent_style   = ParagraphStyle('ent', parent=styles['Heading2'],
                                  fontSize=13, spaceAfter=3,
                                  textColor=colors.HexColor('#1a1a2e'))
    sub_style   = ParagraphStyle('sub', parent=styles['Normal'],
                                  fontSize=10, spaceAfter=2,
                                  textColor=colors.HexColor('#555'))

    elements.append(Paragraph("RÉSUMÉ GLOBAL", title_style))
    elements.append(Paragraph(f"Periode  : {mois:02d}/{annee}", sub_style))
    elements.append(Paragraph(f"Genere le: {date.today().strftime('%d/%m/%Y')}", sub_style))
    elements.append(Spacer(1, 0.5*cm))

    # ─── TABLEAU RÉCAPITULATIF GLOBAL ────────────────────────────────────────
    recap_header = ['Entreprise', 'Voyages', 'Tonnage', 'Recettes', 'Couts', 'Benefice']
    recap_rows   = [recap_header]

    grand_recettes = 0
    grand_couts    = 0
    grand_tonnes   = 0

    for e in entreprises_data:
        recap_rows.append([
            e['nom'],
            str(e['nb_voyages']),
            f"{e['total_tonnes']:.1f} t",
            f"{e['total_montant']:.0f} MRU",
            f"{e['total_couts']:.0f} MRU",
            f"{e['benefice']:.0f} MRU",
        ])
        grand_recettes += e['total_montant']
        grand_couts    += e['total_couts']
        grand_tonnes   += e['total_tonnes']

    grand_benefice = grand_recettes - grand_couts
    recap_rows.append([
        'TOTAL',
        '',
        f"{grand_tonnes:.1f} t",
        f"{grand_recettes:.0f} MRU",
        f"{grand_couts:.0f} MRU",
        f"{grand_benefice:.0f} MRU",
    ])

    recap_table = Table(recap_rows, colWidths=[4*cm, 1.5*cm, 2.5*cm, 3*cm, 3*cm, 3*cm])
    recap_table.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0),  (-1, 0),  colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',      (0, 0),  (-1, 0),  colors.white),
        ('FONTNAME',       (0, 0),  (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0, 0),  (-1, -1), 8),
        ('ALIGN',          (0, 0),  (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1),  (-1, -2), [colors.white, colors.HexColor('#f0f4ff')]),
        ('BACKGROUND',     (0, -1), (-1, -1), colors.HexColor('#c8e6c9')),
        ('FONTNAME',       (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID',           (0, 0),  (-1, -1), 0.5, colors.HexColor('#cccccc')),
    ]))
    elements.append(recap_table)
    elements.append(Spacer(1, 0.8*cm))

    # ─── DETAIL PAR ENTREPRISE ────────────────────────────────────────────────
    for e in entreprises_data:
        elements.append(HRFlowable(width="100%", thickness=1,
                                    color=colors.HexColor('#1a1a2e')))
        elements.append(Spacer(1, 0.2*cm))
        elements.append(Paragraph(f"📋 {e['nom'].upper()}", ent_style))

        header = ['ID', 'Date', 'Camion', 'Tonnage', 'Prix/t', 'Montant']
        rows   = [header]
        for v in e['voyages']:
            montant = v['tonnage'] * v['prix_client']
            rows.append([
                str(v['id']),
                str(v['date']),
                v['num_camion'],
                f"{v['tonnage']:.2f} t",
                f"{v['prix_client']:.0f} MRU",
                f"{montant:.0f} MRU",
            ])
        rows.append(['', '', '', f"{e['total_tonnes']:.2f} t",
                     'TOTAL', f"{e['total_montant']:.0f} MRU"])

        t = Table(rows, colWidths=[1.2*cm, 2.8*cm, 3*cm, 2.5*cm, 2.8*cm, 3.7*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',     (0, 0),  (-1, 0),  colors.HexColor('#334155')),
            ('TEXTCOLOR',      (0, 0),  (-1, 0),  colors.white),
            ('FONTNAME',       (0, 0),  (-1, 0),  'Helvetica-Bold'),
            ('FONTSIZE',       (0, 0),  (-1, -1), 8),
            ('ALIGN',          (0, 0),  (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1),  (-1, -2), [colors.white, colors.HexColor('#f8faff')]),
            ('BACKGROUND',     (0, -1), (-1, -1), colors.HexColor('#e8f5e9')),
            ('FONTNAME',       (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID',           (0, 0),  (-1, -1), 0.4, colors.HexColor('#dddddd')),
        ]))
        elements.append(t)

        # Mini résumé benefice
        bene_data = [
            ['Recettes', f"{e['total_montant']:.0f} MRU"],
            ['Couts',    f"{e['total_couts']:.0f} MRU"],
            ['Benefice', f"{e['benefice']:.0f} MRU"],
        ]
        bt = Table(bene_data, colWidths=[4*cm, 4*cm])
        bt.setStyle(TableStyle([
            ('FONTSIZE',      (0, 0),  (-1, -1), 9),
            ('BACKGROUND',    (0, -1), (-1, -1), colors.HexColor('#c8e6c9')),
            ('FONTNAME',      (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('ALIGN',         (1, 0),  (1, -1),  'RIGHT'),
            ('GRID',          (0, 0),  (-1, -1), 0.3, colors.grey),
            ('TOPPADDING',    (0, 0),  (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0),  (-1, -1), 3),
        ]))
        elements.append(Spacer(1, 0.2*cm))
        elements.append(bt)
        elements.append(Spacer(1, 0.4*cm))

    doc.build(elements)
    return filename
