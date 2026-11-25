from __future__ import annotations

from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
)

from .models import CVDocument


NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#147D78")
INK = colors.HexColor("#24323F")
MUTED = colors.HexColor("#5E6B76")
RULE = colors.HexColor("#D7E0E7")


def _clean(value: str) -> str:
    return escape(
        value.replace("—", "-").replace("–", "-").replace("‑", "-").replace("→", "->")
    )


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "name": ParagraphStyle(
            "CVName", parent=sample["Title"], fontName="Helvetica-Bold", fontSize=22,
            leading=25, textColor=NAVY, alignment=TA_CENTER, spaceAfter=4,
        ),
        "target": ParagraphStyle(
            "CVTarget", parent=sample["Normal"], fontName="Helvetica-Bold", fontSize=10,
            leading=13, textColor=TEAL, alignment=TA_CENTER, spaceAfter=3,
        ),
        "contact": ParagraphStyle(
            "CVContact", parent=sample["Normal"], fontSize=8.2, leading=11,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "CVHeading", parent=sample["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=13, textColor=NAVY, spaceBefore=7, spaceAfter=4, borderColor=TEAL,
            borderWidth=0, borderPadding=0,
        ),
        "body": ParagraphStyle(
            "CVBody", parent=sample["BodyText"], fontSize=8.8, leading=12,
            textColor=INK, spaceAfter=4,
        ),
        "item_title": ParagraphStyle(
            "CVItemTitle", parent=sample["BodyText"], fontName="Helvetica-Bold",
            fontSize=9.2, leading=12, textColor=INK, spaceAfter=1,
        ),
        "meta": ParagraphStyle(
            "CVMeta", parent=sample["BodyText"], fontName="Helvetica-Oblique",
            fontSize=8, leading=10, textColor=MUTED, spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "CVBullet", parent=sample["BodyText"], fontSize=8.6, leading=11.5,
            textColor=INK, leftIndent=8, firstLineIndent=-5, bulletIndent=2, spaceAfter=2,
        ),
    }


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(RULE)
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, 10.5 * mm, f"Target evidence: {doc.target_document_id}")
    canvas.drawRightString(width - 18 * mm, 10.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def cv_pdf_bytes(cv: CVDocument) -> bytes:
    styles = _styles()
    buffer = BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title=f"{cv.profile_name} — {cv.target_title}",
        author=cv.profile_name,
    )
    doc.target_document_id = cv.target_document_id
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="cv", frames=[frame], onPage=_header_footer))

    story = [
        Paragraph(_clean(cv.profile_name), styles["name"]),
        Paragraph(_clean(f"TAILORED FOR {cv.target_title.upper()}"), styles["target"]),
        Paragraph(_clean(cv.contact_line), styles["contact"]),
        Paragraph("PROFESSIONAL SUMMARY", styles["heading"]),
        Paragraph(_clean(cv.professional_summary), styles["body"]),
    ]

    sections = {
        "Matched Skills": lambda: [
            Paragraph("MATCHED SKILLS", styles["heading"]),
            Paragraph(_clean(" • ".join(cv.skills)), styles["body"]),
        ],
        "Projects": lambda: [
            element
            for project in cv.projects
            for element in [
                Paragraph("PROJECTS", styles["heading"]) if project is cv.projects[0] else Spacer(1, 2),
                KeepTogether([
                    Paragraph(_clean(project.title), styles["item_title"]),
                    Paragraph(_clean(project.context), styles["meta"]),
                    Paragraph(_clean(f"• {project.description}"), styles["bullet"]),
                    Paragraph(_clean(f"Technologies: {', '.join(project.technologies)}"), styles["meta"]),
                ]),
            ]
        ] if cv.projects else [],
        "Experience": lambda: [
            element
            for item in cv.experience
            for element in [
                Paragraph("EXPERIENCE", styles["heading"]) if item is cv.experience[0] else Spacer(1, 2),
                KeepTogether([
                    Paragraph(_clean(f"{item.role} - {item.organisation}"), styles["item_title"]),
                    Paragraph(_clean(item.employment_type), styles["meta"]),
                    Paragraph(_clean(f"• {item.description}"), styles["bullet"]),
                ]),
            ]
        ] if cv.experience else [],
        "Education": lambda: [
            element
            for item in cv.education
            for element in [
                Paragraph("EDUCATION", styles["heading"]) if item is cv.education[0] else Spacer(1, 2),
                KeepTogether([
                    Paragraph(_clean(item.qualification), styles["item_title"]),
                    Paragraph(_clean(f"{item.institution} | {item.status}"), styles["meta"]),
                    Paragraph(_clean("Relevant study: " + "; ".join(item.details)), styles["body"])
                    if item.details else Spacer(1, 0),
                ]),
            ]
        ] if cv.education else [],
        "Achievements": lambda: [
            Paragraph("ACHIEVEMENTS & ACTIVITIES", styles["heading"]),
            *[Paragraph(_clean(f"• {item}"), styles["bullet"]) for item in cv.achievements],
        ] if cv.achievements else [],
    }
    for section_name in cv.section_order:
        factory = sections.get(section_name)
        if factory:
            story.extend(factory())
    doc.build(story)
    return buffer.getvalue()


def export_cv_pdf(cv: CVDocument, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv_pdf_bytes(cv))
    return path
