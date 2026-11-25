from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "architecture_diagram.png"


def box(ax, x, y, width, height, title, detail, color):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=color, edgecolor="#17324d", linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.65, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#17324d")
    ax.text(x + width / 2, y + height * 0.31, detail, ha="center", va="center", fontsize=7.3, color="#334c5e", wrap=True)


def arrow(ax, start, end, label=""):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#147d78"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha="center", fontsize=7, color="#526978")


def main():
    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f9fb")
    ax.set_facecolor("#f7f9fb")
    ax.text(0.5, 0.96, "EvidenceCV: confirmed-profile to tailored-PDF data flow", ha="center", fontsize=16, fontweight="bold", color="#17324d")
    ax.text(0.5, 0.915, "User facts and job evidence remain separate until the CV planning step", ha="center", fontsize=9, color="#526978")

    box(ax, 0.04, 0.67, 0.17, 0.15, "1. Introduction", "Natural language\n+ user_id", "#e8f4f5")
    box(ax, 0.28, 0.67, 0.18, 0.15, "2. Profile Agent", "Extract facts\nAsk follow-ups", "#e8f4f5")
    box(ax, 0.54, 0.67, 0.18, 0.15, "3. Review", "User checks summary\nand confirms", "#fff3d9")
    box(ax, 0.79, 0.67, 0.17, 0.15, "4. Profile Store", "Session keyed by user_id\nconfirmed facts only", "#e9eef8")
    arrow(ax, (0.21, 0.745), (0.28, 0.745))
    arrow(ax, (0.46, 0.745), (0.54, 0.745), "draft")
    arrow(ax, (0.72, 0.745), (0.79, 0.745), "confirm")

    box(ax, 0.07, 0.34, 0.21, 0.16, "5. Job RAG", "Embed 9 jobs + query\nCosine top-3 + excerpts", "#e9eef8")
    box(ax, 0.39, 0.34, 0.21, 0.16, "6. Tool Loop", "profile_tool → Observation\nretrieval_tool → Observation", "#e8f4f5")
    box(ax, 0.71, 0.34, 0.21, 0.16, "7. Job Selection", "User selects exact\ndocument_id", "#fff3d9")
    arrow(ax, (0.84, 0.67), (0.55, 0.50), "confirmed state")
    arrow(ax, (0.39, 0.42), (0.28, 0.42), "query")
    arrow(ax, (0.28, 0.39), (0.39, 0.39), "top 3")
    arrow(ax, (0.60, 0.42), (0.71, 0.42))

    box(ax, 0.15, 0.08, 0.22, 0.16, "8. CV Planner", "Select/reorder fact IDs\nMatched vs missing", "#e8f4f5")
    box(ax, 0.48, 0.08, 0.20, 0.16, "9. Editable CV", "Human review\nReconfirm edits", "#fff3d9")
    box(ax, 0.78, 0.08, 0.17, 0.16, "10. PDF", "Styled export\nTarget ID recorded", "#e9eef8")
    arrow(ax, (0.79, 0.34), (0.37, 0.22), "job evidence")
    arrow(ax, (0.80, 0.67), (0.32, 0.24), "profile facts")
    arrow(ax, (0.37, 0.16), (0.48, 0.16))
    arrow(ax, (0.68, 0.16), (0.78, 0.16))

    ax.text(0.03, 0.015, "Safeguards: step limit • cross-user access block • source fact IDs • missing requirements never promoted to skills", fontsize=7.5, color="#526978")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
