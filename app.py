import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import base64
import textwrap
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "data" / "train.csv"
IMAGE_DIR = BASE_DIR / "images"


# =========================================================
# 3. LOAD MODEL
# =========================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()

# ==========================================
# MODEL EVALUATION DATA
# ==========================================

evaluation_df = pd.read_csv("data/train.csv")

evaluation_features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
]

evaluation_target = "SalePrice"

X_eval = evaluation_df[evaluation_features].copy()
y_eval = evaluation_df[evaluation_target]

# Handle missing values exactly as in model.py
X_eval["GarageCars"] = X_eval["GarageCars"].fillna(0)
X_eval["TotalBsmtSF"] = X_eval["TotalBsmtSF"].fillna(0)

# Same split used when training the model
_, X_test, _, y_test = train_test_split(
    X_eval,
    y_eval,
    test_size=0.2,
    random_state=42
)

# Predictions on unseen test data
y_pred = model.predict(X_test)


# ==========================================
# MODEL FEATURE IMPORTANCE
# ==========================================

FEATURE_NAMES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
]

feature_importance = dict(
    zip(FEATURE_NAMES, model.feature_importances_)
)


# =========================================================
# 4. LOAD DATASET
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("Unable to load the training dataset.")
    st.exception(e)
    st.stop()


# =========================================================
# 5. CUSTOM CSS
# =========================================================

CSS_PATH = BASE_DIR / "styles" / "style.css"

with open(CSS_PATH, "r") as f:
    css = f.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)


# =========================================================
# PDF PREDICTION REPORT
# =========================================================

def generate_prediction_report(
    prediction,
    lower_bound,
    upper_bound,
    overall_qual,
    living_area,
    garage_cars,
    basement_area,
    full_bath,
    year_built,
    feature_importance
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#172036"),
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontSize=16,
        leading=22,
        textColor=colors.HexColor("#172036"),
        spaceBefore=18,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontSize=10,
        leading=16,
        textColor=colors.HexColor("#475569")
    )

    price_style = ParagraphStyle(
        "Price",
        parent=styles["Heading1"],
        fontSize=28,
        leading=34,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f766e"),
        spaceAfter=10
    )

    story = []

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "HOUSE PRICE PREDICTOR",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Machine Learning Property Valuation Report",
            subtitle_style
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#e2e8f0"),
            spaceAfter=20
        )
    )

    # -----------------------------------------------------
    # ESTIMATED VALUE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "ESTIMATED PROPERTY VALUE",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"${prediction:,.2f}",
            price_style
        )
    )

    story.append(
        Paragraph(
            f"Approximate estimated range: "
            f"<b>${lower_bound:,.0f}</b> — "
            f"<b>${upper_bound:,.0f}</b>",
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    # -----------------------------------------------------
    # PROPERTY DETAILS
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Property Details",
            heading_style
        )
    )

    property_data = [
        ["Property Feature", "Value"],
        ["Overall Quality", f"{overall_qual}/10"],
        ["Above-Ground Living Area", f"{living_area:,} sq ft"],
        ["Basement Area", f"{basement_area:,} sq ft"],
        ["Full Bathrooms", str(full_bath)],
        ["Garage Capacity", f"{garage_cars} car(s)"],
        ["Year Built", str(year_built)]
    ]

    property_table = Table(
        property_data,
        colWidths=[3.5 * inch, 2.2 * inch]
    )

    property_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172036")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "FONTNAME",
                (0, 1),
                (-1, -1),
                "Helvetica"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#e2e8f0")
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#f8fafc")
            ),
            (
                "TEXTCOLOR",
                (0, 1),
                (-1, -1),
                colors.HexColor("#334155")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(property_table)

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "What Influences the Prediction?",
            heading_style
        )
    )

    feature_data = [
        ["Feature", "Importance"]
    ]

    feature_names = {
        "OverallQual": "Overall Quality",
        "GrLivArea": "Living Area",
        "GarageCars": "Garage Capacity",
        "TotalBsmtSF": "Basement Area",
        "FullBath": "Full Bathrooms",
        "YearBuilt": "Year Built"
    }

    for feature, importance in feature_importance.items():

        display_name = feature_names.get(
            feature,
            feature
        )

        feature_data.append([
            display_name,
            f"{importance * 100:.1f}%"
        ])

    feature_table = Table(
        feature_data,
        colWidths=[3.5 * inch, 2.2 * inch]
    )

    feature_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172036")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#e2e8f0")
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#f8fafc")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(feature_table)

    # -----------------------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Model Performance",
            heading_style
        )
    )

    performance_data = [
        ["Metric", "Result"],
        ["Model", "Random Forest"],
        ["R² Score", "88.9%"],
        ["Mean Absolute Error", "$19,102.98"],
        ["Root Mean Squared Error", "$29,191.38"]
    ]

    performance_table = Table(
        performance_data,
        colWidths=[3.5 * inch, 2.2 * inch]
    )

    performance_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172036")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.HexColor("#e2e8f0")
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#f8fafc")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    story.append(performance_table)

    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    story.append(
        Paragraph(
            "Important Disclaimer",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "This estimate is generated by a machine learning model "
            "and is intended for informational purposes only. Actual "
            "property values may vary depending on location, market "
            "conditions, property condition and other factors not "
            "included in the model.",
            normal_style
        )
    )

    story.append(Spacer(1, 20))

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    generated_date = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    story.append(
        Paragraph(
            f"Report generated: {generated_date}",
            normal_style
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "Built with Python • Pandas • Scikit-learn • Streamlit",
            subtitle_style
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer

# =========================================================
# 6. HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>House Price Predictor</h1>
        <p>
            Estimate the potential market value of a property
            using a machine learning model trained on real
            housing data.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 7. AUTOMATIC PROPERTY CAROUSEL
# =========================================================

st.markdown(
    '<div class="section-title">Featured Properties</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# PROPERTY TITLES
# ---------------------------------------------------------

house_titles = [
    "Modern Family Home",
    "Contemporary Residence",
    "Luxury Family House",
    "Elegant Modern Home",
    "Premium Residence"
]


# ---------------------------------------------------------
# PROPERTY DESCRIPTIONS
# ---------------------------------------------------------

house_descriptions = [
    "Spacious • Contemporary • Family Friendly",
    "Modern • Comfortable • Stylish",
    "Luxury • Spacious • Premium",
    "Elegant • Modern • Peaceful",
    "Premium • Contemporary • Spacious"
]


# ---------------------------------------------------------
# PROPERTY IMAGES
# ---------------------------------------------------------

image_files = [
    IMAGE_DIR / "house1.jpg",
    IMAGE_DIR / "house2.jpg",
    IMAGE_DIR / "house3.jpg",
    IMAGE_DIR / "house4.jpg",
    IMAGE_DIR / "house5.jpg"
]


# =========================================================
# BUILD SLIDES
# =========================================================

slides = []

for image_path, title, description in zip(
    image_files,
    house_titles,
    house_descriptions
):

    if not image_path.exists():
        continue

    with open(image_path, "rb") as image_file:

        image_base64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")


    slide = f"""
<div class="carousel-slide">
    <img
        src="data:image/jpeg;base64,{image_base64}"
        alt="{title}"
    >
    <div class="carousel-overlay">
        <div class="carousel-text">
            <div class="property-badge">
                FEATURED PROPERTY
            </div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
    </div>
</div>
"""
    slides.append(slide)


# =========================================================
# DISPLAY CAROUSEL
# =========================================================

if slides:

    total_seconds = len(slides) * 4

    carousel_html = f"""
<style>

.carousel {{
    position: relative;
    width: 100%;
    height: 520px;
    overflow: hidden;
    border-radius: 26px;
    background: #172033;
    box-shadow: 0 18px 45px rgba(20, 30, 50, 0.16);
}}

.carousel-slide {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    opacity: 0;

    animation:
        houseCarousel {total_seconds}s
        infinite ease-in-out;
}}

.carousel-slide img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}}

.carousel-overlay {{
    position: absolute;
    inset: 0;

    display: flex;
    align-items: flex-end;

    padding: 45px;

    background:
        linear-gradient(
            to top,
            rgba(0, 0, 0, 0.85) 0%,
            rgba(0, 0, 0, 0.55) 30%,
            rgba(0, 0, 0, 0.12) 70%,
            rgba(0, 0, 0, 0) 100%
        );

    z-index: 2;
}}

.carousel-text {{
    color: white;
    max-width: 750px;
    text-align: left;
}}

.property-badge {{
    display: inline-block;

    padding: 7px 14px;
    margin-bottom: 12px;

    border-radius: 50px;

    background: #ff4b4b;
    color: white;

    font-size: 0.75rem;
    font-weight: 800;

    letter-spacing: 1px;
}}

.carousel-text h2 {{
    margin: 0 0 8px 0;

    color: white !important;

    font-size: 2.7rem;
    font-weight: 800;

    line-height: 1.1;

    text-shadow:
        0 2px 8px rgba(0, 0, 0, 0.55);
}}

.carousel-text p {{
    margin: 0;

    color: rgba(255, 255, 255, 0.95) !important;

    font-size: 1.05rem;
    font-weight: 500;

    text-shadow:
        0 2px 6px rgba(0, 0, 0, 0.5);
}}


/* =====================================================
   SLIDE DELAYS
   ===================================================== */

.carousel-slide:nth-child(2) {{
    animation-delay: 4s;
}}

.carousel-slide:nth-child(3) {{
    animation-delay: 8s;
}}

.carousel-slide:nth-child(4) {{
    animation-delay: 12s;
}}

.carousel-slide:nth-child(5) {{
    animation-delay: 16s;
}}


/* =====================================================
   CAROUSEL ANIMATION
   ===================================================== */

@keyframes houseCarousel {{

    0% {{
        opacity: 0;
    }}

    4% {{
        opacity: 1;
    }}

    18% {{
        opacity: 1;
    }}

    22% {{
        opacity: 0;
    }}

    100% {{
        opacity: 0;
    }}

}}


/* =====================================================
   MOBILE
   ===================================================== */

@media (max-width: 768px) {{

    .carousel {{
        height: 390px;
        border-radius: 20px;
    }}

    .carousel-overlay {{
        padding: 25px;
    }}

    .carousel-text h2 {{
        font-size: 1.8rem;
    }}

    .carousel-text p {{
        font-size: 0.9rem;
    }}

}}

</style>

<div class="carousel">

    {"".join(slides)}

</div>
"""

    # IMPORTANT:
    # Remove indentation from the HTML before sending
    # it to Streamlit's Markdown renderer.

    carousel_html = textwrap.dedent(
        carousel_html
    ).strip()


    st.markdown(
        carousel_html,
        unsafe_allow_html=True
    )

else:

    st.warning(
        "No property images were found. "
        "Make sure house1.jpg through house5.jpg "
        "are inside the project's images folder."
    )


# =========================================================
# 8. PROPERTY INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">Property Information</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="input-info">
        <div class="input-info-title">
            Tell us about the property
        </div>
        <div class="input-info-text">
            Enter the details below. The machine learning
            model will use these characteristics to estimate
            the property's value.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 9. PROPERTY INPUTS
# =========================================================

col1, col2 = st.columns(2)


with col1:

    overall_qual = st.slider(
        "Overall Quality of the House (1–10)",
        min_value=1,
        max_value=10,
        value=7,
        help="1 = Very poor quality • 10 = Excellent quality"
    )


    living_area = st.number_input(
        "Above-Ground Living Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=2000,
        step=100,
        help="Enter the total above-ground living area."
    )


    garage_cars = st.number_input(
        "Garage Capacity (cars)",
        min_value=0,
        max_value=6,
        value=2,
        step=1,
        help="Enter the number of cars the garage can accommodate."
    )


with col2:

    basement_area = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=100,
        help="Enter the total basement area."
    )


    full_bath = st.number_input(
        "Number of Full Bathrooms",
        min_value=0,
        max_value=8,
        value=2,
        step=1,
        help="Enter the number of full bathrooms."
    )


    year_built = st.number_input(
        "Year the House Was Built",
        min_value=1800,
        max_value=2026,
        value=2005,
        step=1,
        help="Enter the year construction was completed."
    )


# =========================================================
# 10. PREDICT BUTTON
# =========================================================

predict_button = st.button(
    "Predict House Price",
    use_container_width=False
)


# =========================================================
# 11. MAKE PREDICTION
# =========================================================

if predict_button:
  

    try:

        # -------------------------------------------------
        # Create dataframe using the EXACT six features
        # used by the trained Random Forest model.
        # -------------------------------------------------

        house = pd.DataFrame({

            "OverallQual": [overall_qual],

            "GrLivArea": [living_area],

            "GarageCars": [garage_cars],

            "TotalBsmtSF": [basement_area],

            "FullBath": [full_bath],

            "YearBuilt": [year_built]

        })


        # -------------------------------------------------
        # Make prediction
        # -------------------------------------------------

        prediction = model.predict(house)[0]

        # =========================================================
        # ESTIMATED PRICE RANGE
        # =========================================================
    
        rmse = 29191.38
    
        lower_bound = max(0, prediction - rmse)
        upper_bound = prediction + rmse


        # =================================================
        # 12. DISPLAY PREDICTION
        # =================================================

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    ESTIMATED PROPERTY VALUE
                </div>
                <div class="prediction-price">
                    ${prediction:,.2f}
                </div>
                <div class="prediction-description">
                    Estimated based on the property characteristics
                    you provided.
                </div>
                <div class="prediction-range">
                    <div class="range-label">
                        Approximate Estimated Range
                    </div>
                    <div class="range-value">
                        ${lower_bound:,.0f}
                        <span>—</span>
                        ${upper_bound:,.0f}
                    </div>
                    <div class="range-description">
                        This range reflects the model's historical
                        prediction error based on the RMSE observed
                        during evaluation.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # =========================================================
        # DOWNLOAD PREDICTION REPORT
        # =========================================================



        pdf_report = generate_prediction_report(
            prediction=prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            overall_qual=overall_qual,
            living_area=living_area,
            garage_cars=garage_cars,
            basement_area=basement_area,
            full_bath=full_bath,
            year_built=year_built,
            feature_importance=feature_importance
        )

        st.download_button(
            label="📄 Download Prediction Report",
            data=pdf_report,
            file_name="house_price_prediction_report.pdf",
            mime="application/pdf"
        )

        # =================================================
        # 13. PROPERTY SUMMARY
        # =================================================

        st.markdown(
            """
            <div class="summary-card">
                <div class="summary-title">
                    Property Summary
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        summary_col1, summary_col2, summary_col3 = st.columns(3)


        # -------------------------------------------------
        # COLUMN 1
        # -------------------------------------------------

        with summary_col1:

            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-label">
                        Overall Quality
                    </div>
                    <div class="summary-value">
                        {overall_qual}/10
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-label">
                        Living Area
                    </div>
                    <div class="summary-value">
                        {living_area:,} sq ft
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # COLUMN 2
        # -------------------------------------------------

        with summary_col2:

            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-label">
                        Garage
                    </div>
                    <div class="summary-value">
                        {garage_cars} car(s)
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-label">
                        Basement
                    </div>
                    <div class="summary-value">
                        {basement_area:,} sq ft
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # COLUMN 3
        # -------------------------------------------------

        with summary_col3:

            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-label">
                        Bathrooms
                    </div>
                    <div class="summary-value">
                        {full_bath}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="summary-item">
                    <div class="summary-label">
                        Year Built
                    </div>
                    <div class="summary-value">
                        {year_built}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # 14. MODEL INFORMATION
        # =================================================

        st.info(
            "Random Forest generated this estimate using "
            "6 key property features."
        )

    except Exception as e:
        st.error(f"An error occurred while generating the prediction: {e}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    """
    <h2>What Influences the Prediction?</h2>
    <p style="font-size:18px; color:#64748b;">
    The chart below shows how important each property characteristic
    was to the Random Forest model.
    </p>
    """,
    unsafe_allow_html=True
)

# Convert model feature importance to percentages
importance_df = pd.DataFrame({
    "Feature": [
        "Overall Quality",
        "Living Area",
        "Garage Capacity",
        "Basement Area",
        "Full Bathrooms",
        "Year Built"
    ],
    "Importance": [
        feature_importance["OverallQual"],
        feature_importance["GrLivArea"],
        feature_importance["GarageCars"],
        feature_importance["TotalBsmtSF"],
        feature_importance["FullBath"],
        feature_importance["YearBuilt"]
    ]
})

# Convert to percentage
importance_df["Importance"] = importance_df["Importance"] * 100

# Sort for horizontal bar chart
importance_df = importance_df.sort_values("Importance")

# Create chart
fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="#1f679b"
)

# Add percentage labels
for bar, value in zip(bars, importance_df["Importance"]):
    ax.text(
        value + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

ax.set_xlabel("Importance (%)")
ax.set_xlim(0, max(importance_df["Importance"]) + 8)
ax.set_title("Random Forest Feature Importance", fontsize=16, fontweight="bold", pad = 15)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

st.pyplot(fig)

st.info(
    "Higher values indicate that the feature had a greater influence "
    "on the Random Forest model's predictions."
)

# Model explanation
st.markdown(
    """
    <div class="model-performance-info">
        <h3>What do these numbers mean?</h3>
        <p>
            <strong>R² Score (88.9%)</strong> indicates that the model
            explains approximately 88.9% of the variation in property
            prices within the test data.
        </p>
        <p>
            <strong>MAE ($19,103)</strong> means that the model's
            predictions are off by approximately $19,103 on average.
        </p>
        <p>
            <strong>RMSE ($29,191)</strong> gives more weight to larger
            prediction errors and provides an indication of how much
            those larger errors affect the model.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ACTUAL VS PREDICTED PRICES
# ============================================================

st.markdown("""
<div class="section-title">
    Actual vs Predicted Prices
</div>

<p class="section-description">
    This chart compares the actual property prices from the test dataset
    with the prices predicted by the Random Forest model.
</p>
""", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

# Perfect prediction line
min_price = min(y_test.min(), y_pred.min())
max_price = max(y_test.max(), y_pred.max())

ax.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

ax.set_xlabel("Actual House Price")
ax.set_ylabel("Predicted House Price")
ax.set_title("Actual vs Predicted House Prices")

ax.grid(alpha=0.2)

st.pyplot(fig)

st.info(
    "Points closer to the diagonal line represent predictions that are "
    "closer to the actual property prices."
)

# =================================================
# 15. DISCLAIMER
# =================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important:</strong>
        This estimate is generated by a machine
        learning model and is intended for
        informational purposes only.

        Actual property values may vary depending
        on location, market conditions, property
        condition and other factors not included
        in the model.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 16. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <div class="footer-title">
            House Price Predictor
        </div>
        <div class="footer-text">
            Machine Learning Project
            <br>
            Built with Python, Pandas, Scikit-learn and Streamlit
        </div>
    </div>
    """,
    unsafe_allow_html=True
)