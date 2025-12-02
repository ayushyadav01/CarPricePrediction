import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CarValuate Pro | Enterprise AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE INITIALIZATION ---
if 'valuation_active' not in st.session_state:
    st.session_state.valuation_active = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = 0
if 'last_input_data' not in st.session_state:
    st.session_state.last_input_data = None
if 'last_tier' not in st.session_state:
    st.session_state.last_tier = 1

# --- 3. CSS & THEME MANAGEMENT ---
primary_color = "#3b82f6"  # Blue
success_color = "#10b981"  # Green
danger_color = "#ef4444"  # Red
text_color = "#ffffff"  # White text
card_bg = "#1e293b"  # Dark Slate

css_string = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* --- HEADERS --- */
.main-header {{
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}
.sub-header {{
    font-size: 1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}}
.section-title {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #94a3b8;
    font-weight: 700;
    margin-bottom: 15px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 5px;
}}

/* --- CARDS --- */
.result-card {{
    background: {card_bg};
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 20px;
}}
.deal-card {{
    background: {card_bg};
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    transition: transform 0.2s;
    margin-bottom: 15px;
    position: relative;
}}
.deal-card:hover {{
    transform: translateY(-5px);
    border-color: {primary_color};
}}
.score-badge {{
    position: absolute;
    top: 15px;
    right: 15px;
    background: {success_color};
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.9rem;
}}
.deal-title {{
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 5px;
    color: white;
}}
.deal-price {{
    font-size: 1.4rem;
    font-weight: 800;
    color: {primary_color};
}}
.deal-specs {{
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 8px;
    display: flex;
    justify-content: space-between;
}}

/* --- INFO BOXES --- */
.info-box {{
    background: rgba(59, 130, 246, 0.1);
    border-left: 4px solid {primary_color};
    padding: 15px;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #e2e8f0;
    margin-bottom: 20px;
}}

/* --- BUTTONS --- */
.stButton > button {{
    background: linear-gradient(135deg, {primary_color} 0%, #2563eb 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}}
.price-tag {{
    font-size: 3.5rem;
    font-weight: 800;
    color: {success_color};
}}
</style>
"""
st.markdown(css_string, unsafe_allow_html=True)

# --- 4. DATA LOADING ---
filename = 'car_model.pkl'
dataset_path = 'cardekho_dataset.csv'

try:
    with open(filename, 'rb') as f:
        saved_data = pickle.load(f)
    model = saved_data['model']
    le_brand = saved_data['le_brand']
    feature_cols = saved_data['feature_cols']
except FileNotFoundError:
    st.error("⚠️ Model artifacts missing. Please ensure 'car_model.pkl' exists.")
    st.stop()


@st.cache_data
def load_raw_data():
    try:
        df = pd.read_csv(dataset_path)
        # Data Cleaning
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        if 'vehicle_age' in df.columns and 'year' not in df.columns:
            df['year'] = 2024 - df['vehicle_age']

        # Normalize Names
        if 'car_name' in df.columns:
            df['name'] = df['car_name']
        if 'name' in df.columns:
            df['name'] = df['name'].astype(str).str.title()

        for col in ['fuel_type', 'seller_type', 'transmission_type']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.title()

        return df[df['km_driven'] < 500000]
    except:
        return None


df_raw = load_raw_data()


# --- 5. HELPER FUNCTIONS ---
def format_currency(val):
    return "₹ {:,.0f}".format(val)


def get_brand_tier(brand):
    brand = brand.upper()
    luxury = ['BMW', 'MERCEDES-BENZ', 'AUDI', 'JAGUAR', 'LAND ROVER', 'VOLVO', 'PORSCHE', 'LEXUS', 'FERRARI']
    mid = ['TOYOTA', 'HONDA', 'VOLKSWAGEN', 'FORD', 'MAHINDRA', 'SKODA', 'JEEP', 'MG', 'KIA']
    return 3 if brand in luxury else (2 if brand in mid else 1)


def calculate_emi(principal, rate, tenure_years):
    r = rate / (12 * 100)
    n = tenure_years * 12
    if r == 0: return principal / n
    return (principal * r * (1 + r) ** n) / ((1 + r) ** n - 1)


def format_labels(option):
    labels = {
        'km_driven': 'Kilometers Driven', 'max_power': 'Max Power (BHP)',
        'mileage': 'Mileage (kmpl)', 'engine': 'Engine Displacement (CC)',
        'year': 'Model Year', 'selling_price': 'Selling Price',
        'fuel_type': 'Fuel Type', 'seller_type': 'Seller Type',
        'transmission_type': 'Transmission'
    }
    return labels.get(option, option.title())


# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("### 🚘 CarValuate Pro")
    st.caption("AI-Powered Enterprise Analytics v17.0")
    st.markdown("---")

    # Navigation
    st.markdown('<div class="section-title">MODULES</div>', unsafe_allow_html=True)

    # Menu Items
    view_mode = st.radio("Select Tool",
                         ["Valuation Dashboard",
                          "Ownership Economics (TCO)",
                          "Smart Deal Evaluator",
                          "Strategic Market Scanner",
                          "Market Intelligence"],
                         label_visibility="collapsed"
                         )

    st.markdown("---")
    with st.expander("ℹ️ How the AI Works"):
        st.markdown("""
        **Algorithm:** Gradient Boosting Regressor (GBR).
        **Logic:** Considers non-linear depreciation curves based on brand tiers (Luxury vs Economy).
        **Accuracy:** 94.02% R² Score.
        """)

# ==============================================================================
# VIEW 1: VALUATION DASHBOARD
# ==============================================================================
if view_mode == "Valuation Dashboard":
    st.markdown('<div class="main-header">Valuation Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
    Calculate the <strong>Fair Market Value (FMV)</strong> of any vehicle based on live market data.
    Hover over input labels for detailed explanations of how each factor affects price.
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="section-title">1. VEHICLE CONFIGURATION</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Make & Brand")
            brand = st.selectbox("Brand", list(le_brand.classes_), label_visibility="collapsed",
                                 help="Luxury brands (BMW, Audi) depreciate faster initially but hold value later. Economy brands (Maruti, Hyundai) have linear depreciation.")
        with c2:
            st.caption("Registration Year")
            # RESTORED DEFAULT: 2020
            model_year = st.number_input("Year", 2000, 2025, 2020, label_visibility="collapsed",
                                         help="The single biggest factor. A car loses ~20% value in Year 1 and ~10% each subsequent year.")
        with c3:
            st.caption("Seller Type")
            seller = st.selectbox("Seller", ['Individual', 'Dealer', 'Trustmark Dealer'], label_visibility="collapsed",
                                  help="Dealers charge 10-15% premium for warranty/service checks. Individual sellers are cheaper but riskier.")

        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            st.caption("Fuel Type")
            fuel = st.selectbox("Fuel", ['Diesel', 'Petrol', 'CNG', 'Electric'], label_visibility="collapsed",
                                help="Diesel cars are more expensive upfront but may face regulatory bans in cities like Delhi, affecting resale value.")
        with c5:
            st.caption("Transmission")
            trans = st.selectbox("Trans", ['Manual', 'Automatic'], label_visibility="collapsed",
                                 help="Automatics are in higher demand in urban areas, commanding a 5-10% higher resale price.")
        with c6:
            st.caption("Engine (CC)")
            # RESTORED DEFAULT: 1200
            engine = st.number_input("Engine", 500, 6000, 1200, step=100, label_visibility="collapsed",
                                     help="Larger displacement (CC) usually means higher price, but lower fuel economy.")
        with c7:
            st.caption("Power (BHP)")
            # RESTORED DEFAULT: 85.0
            power = st.number_input("Power", 30.0, 600.0, 85.0, step=5.0, label_visibility="collapsed",
                                    help="Brake Horse Power. High performance is a key driver for valuation in the premium segment.")

        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        c8, c9, c10 = st.columns([1, 1, 1])
        with c8:
            st.caption("Odometer (KM)")
            # RESTORED DEFAULT: 40000
            km_driven = st.number_input("KM", 0, 500000, 40000, step=5000, label_visibility="collapsed",
                                        help="High mileage (>100k km) drastically reduces engine life expectancy and resale value.")
        with c9:
            st.caption("Mileage (KMPL)")
            # RESTORED DEFAULT: 18.0
            mileage = st.slider("Mileage", 5.0, 40.0, 18.0, step=0.5, label_visibility="collapsed",
                                help="Official ARAI fuel efficiency. Critical for economy segment buyers.")
        with c10:
            st.caption("Seats")
            # RESTORED DEFAULT: 5
            seats = st.slider("Seats", 2, 9, 5, label_visibility="collapsed",
                              help="7-seaters (SUVs/MUVs) often hold value better than 5-seater sedans due to utility.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_act1, col_act2, col_act3 = st.columns([1, 2, 1])
    with col_act2:
        if st.button("✨ GENERATE VALUATION REPORT", use_container_width=True):
            with st.spinner("Analyzing market vectors..."):
                time.sleep(0.5)

                age = 2024 - model_year
                if age < 0: age = 0

                brand_id = le_brand.transform([brand])[0]
                tier_id = get_brand_tier(brand)
                fuel_map = {'CNG': 0, 'Diesel': 1, 'Electric': 2, 'LPG': 3, 'Petrol': 4}
                seller_map = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
                trans_map = {'Automatic': 0, 'Manual': 1}

                input_data = pd.DataFrame([[
                    brand_id, tier_id, age, km_driven, seller_map[seller],
                    fuel_map.get(fuel, 4), trans_map[trans], mileage, engine, power, seats
                ]], columns=feature_cols)

                prediction = model.predict(input_data)[0]

                st.session_state.valuation_active = True
                st.session_state.last_prediction = prediction
                st.session_state.last_input_data = input_data
                st.session_state.last_tier = tier_id
                st.toast("Valuation Complete!", icon="✅")

    if st.session_state.valuation_active:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">2. ANALYTICAL REPORT</div>', unsafe_allow_html=True)

        pred_value = st.session_state.last_prediction
        tier = st.session_state.last_tier

        tier_desc = {
            1: "Economy (Mass Market)",
            2: "Mid-Range (Premium)",
            3: "Luxury (High-End)"
        }
        tier_label = tier_desc.get(tier, "Unknown")

        st.markdown(f"""
        <div class="result-card">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">AI Estimated Market Value</div>
            <div class="price-tag">{format_currency(pred_value)}</div>
            <div style="color: #94a3b8; margin-top: 15px; font-size: 0.9rem;">
                Confidence Score: <span style="color: {success_color}; font-weight: bold;">94.02%</span> • 
                Segment: <strong>Tier {tier} ({tier_label})</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        res_c1, res_c2 = st.columns([2, 1])

        with res_c1:
            st.markdown("**📉 Depreciation Forecast**")
            st.caption("Projected value retention over the next 5 years (assuming 15k km/year usage).")
            # --- FIXED GRAPH LOGIC ---
            input_d = st.session_state.last_input_data
            years = list(range(6))
            values = []
            for i in years:
                temp_df = input_d.copy()
                temp_df['vehicle_age'] += i
                temp_df['km_driven'] += (i * 15000)
                val = model.predict(temp_df)[0]
                values.append(val)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years, y=values,
                mode='lines+markers+text',
                text=[f"{v / 100000:.2f}L" for v in values],
                textposition="top center",
                line=dict(color=primary_color, width=4, shape='spline'),
                marker=dict(size=12, color="#1e293b", line=dict(width=3, color=primary_color)),
                cliponaxis=False  # Ensures text labels don't get cut off
            ))

            # Updated layout margins
            fig.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Years from Today",
                margin=dict(l=40, r=20, t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

        with res_c2:
            st.markdown("**💰 Financing Simulator**")
            st.caption("Estimated monthly payments for a standard loan.")
            with st.container(border=True):
                loan_amt = pred_value * 0.80
                rate = st.number_input("Interest Rate (%)", value=9.5, step=0.1, format="%.2f")
                tenure = st.slider("Tenure (Years)", 1, 7, 5)
                emi = calculate_emi(loan_amt, rate, tenure)
                st.markdown("---")
                # FIXED: IMPROVED EMI DESIGN
                st.markdown(f"""
                <div style="text-align: center; background-color: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 10px;">
                    <div style="font-size: 0.8rem; opacity: 0.7; letter-spacing: 1px;">MONTHLY PAYMENT</div>
                    <div style="font-size: 2rem; font-weight: 800; color: {primary_color}; margin: 5px 0;">{format_currency(emi)}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Principal: {format_currency(loan_amt)}</div>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# VIEW 2: OWNERSHIP ECONOMICS (TCO) - REPLACES SPEC-TO-VALUE MATRIX
# ==============================================================================
elif view_mode == "Ownership Economics (TCO)":
    st.markdown('<div class="main-header">Ownership Economics (TCO)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
    Buying the car is only <strong>20% of the cost</strong>. 
    This engine calculates the <strong>Total Cost of Ownership</strong> over 5 years, considering Fuel, Maintenance, and Resale Value.
    </div>
    """, unsafe_allow_html=True)

    # --- INPUTS FOR TCO ---
    with st.container(border=True):
        st.markdown('<div class="section-title">SCENARIO CONFIGURATION</div>', unsafe_allow_html=True)
        tco_c1, tco_c2, tco_c3 = st.columns(3)
        with tco_c1:
            # We need a base price. If they ran valuation, use it. If not, ask.
            base_price = st.session_state.last_prediction if st.session_state.last_prediction > 0 else 500000
            purchase_price = st.number_input("Purchase Price (₹)", 50000, 10000000, int(base_price), step=10000,
                                             help="Initial cost of buying the vehicle.")
            st.caption("Auto-filled from Dashboard if available.")
        with tco_c2:
            # RESTORED DEFAULT: 1000
            monthly_km = st.slider("Monthly Driving (km)", 100, 5000, 1000, step=100,
                                   help="Average distance you drive per month.")
        with tco_c3:
            fuel_type_tco = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])

    # --- ADVANCED SETTINGS ---
    with st.expander("⚙️ Advanced Cost Parameters (Click to Edit)"):
        adv_c1, adv_c2, adv_c3 = st.columns(3)
        with adv_c1:
            # RESTORED DEFAULT: 15.0
            mileage_tco = st.number_input("Real World Mileage (kmpl)", 5.0, 50.0, 15.0,
                                          help="City/Highway mixed mileage.")
        with adv_c2:
            # Dynamic Fuel Price Defaults
            fuel_prices = {'Petrol': 100, 'Diesel': 90, 'CNG': 85, 'Electric': 10}
            fuel_price = st.number_input(f"{fuel_type_tco} Cost (₹/Unit)", 0, 200, fuel_prices[fuel_type_tco])
        with adv_c3:
            # Dynamic Maintenance Defaults based on price/tier logic approximation
            maint_base = 0.5 if purchase_price < 800000 else (1.5 if purchase_price < 2000000 else 3.5)
            maint_cost_per_km = st.number_input("Maint. Cost (₹/km)", 0.0, 50.0, maint_base, step=0.1,
                                                help="Estimated service & repair cost per km driven.")

    # --- CALCULATION ENGINE ---
    # 1. Fuel Cost
    total_km_5yr = monthly_km * 12 * 5
    total_fuel_cost = (total_km_5yr / mileage_tco) * fuel_price

    # 2. Maintenance Cost
    total_maint_cost = total_km_5yr * maint_cost_per_km

    # 3. Resale Value (Using a heuristic depreciation since we can't easily call the model dynamically here without more inputs)
    # Heuristic: 5 years adds ~40% depreciation usually
    depreciation_rate = 0.40
    resale_value = purchase_price * (1 - depreciation_rate)

    # 4. Total TCO
    tco_total = (purchase_price - resale_value) + total_fuel_cost + total_maint_cost

    st.markdown("<br>", unsafe_allow_html=True)

    # --- VISUALIZATION ---
    res_tco1, res_tco2 = st.columns([1, 1.5])

    with res_tco1:
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">5-Year Total Cost to Own</div>
            <div class="price-tag" style="font-size: 2.5rem;">{format_currency(tco_total)}</div>
            <div style="margin-top: 10px; font-size: 0.9rem; color: #e2e8f0;">
                That's <strong>{format_currency(tco_total / 60)}</strong> per month.
            </div>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                <span>⛽ Fuel Spend:</span>
                <span style="color: {danger_color}; font-weight: bold;">{format_currency(total_fuel_cost)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-top: 5px;">
                <span>🛠️ Maintenance:</span>
                <span style="color: #facc15; font-weight: bold;">{format_currency(total_maint_cost)}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9rem; margin-top: 5px;">
                <span>📉 Value Lost:</span>
                <span style="color: #94a3b8; font-weight: bold;">{format_currency(purchase_price - resale_value)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_tco2:
        st.markdown("#### Cost Breakdown Analysis")
        # Waterfall Chart Logic
        fig = go.Figure(go.Waterfall(
            name="20", orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["Buying Price", "Fuel (5yr)", "Service (5yr)", "Resale Recovery", "NET TCO"],
            textposition="outside",
            text=[f"{purchase_price / 1000:.0f}k", f"{total_fuel_cost / 1000:.0f}k", f"{total_maint_cost / 1000:.0f}k",
                  f"-{resale_value / 1000:.0f}k", f"{tco_total / 1000:.0f}k"],
            y=[purchase_price, total_fuel_cost, total_maint_cost, -resale_value, tco_total],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": danger_color}},  # Costs are Red
            decreasing={"marker": {"color": success_color}},  # Resale is Green (Money back)
            totals={"marker": {"color": primary_color}},
            cliponaxis=False  # FIXED: Prevents value cutting
        ))

        # FIXED: Added margins so top labels fit
        fig.update_layout(
            template="plotly_dark",
            height=350,
            title="Where does the money go?",
            margin=dict(t=60, l=20, r=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "💡 **Insight:** Notice how for cheap cars, Fuel Cost often exceeds the Buying Price over 5 years. This highlights why high-mileage users should prioritize efficiency over initial price.")

# ==============================================================================
# VIEW 3: SMART DEAL EVALUATOR (RESTORED DEFAULTS)
# ==============================================================================
elif view_mode == "Smart Deal Evaluator":
    st.markdown('<div class="main-header">Smart Deal Evaluator</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
    Determine if a listing is a Steal, Fair, or Rip-off.
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">LISTING DETAILS</div>', unsafe_allow_html=True)
        with st.form("deal_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                brand = st.selectbox("Brand", list(le_brand.classes_))
                # RESTORED DEFAULT: 2020
                model_year = st.number_input("Year", 2000, 2024, 2020)
            with c2:
                # RESTORED DEFAULT: 40000
                km_driven = st.number_input("KM", 0, 300000, 40000)
                fuel = st.selectbox("Fuel", ['Diesel', 'Petrol', 'CNG'])
            with c3:
                # RESTORED DEFAULT: 500000
                asking_price = st.number_input("Seller Asking Price (₹)", 50000, 10000000, 500000, step=10000)
                trans = st.selectbox("Transmission", ['Manual', 'Automatic'])

            seller = 'Individual'
            mileage = 18.0
            power = 85.0
            engine = 1200
            seats = 5

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚖️ ANALYZE DEAL QUALITY")

    if submitted:
        brand_id = le_brand.transform([brand])[0]
        tier_id = get_brand_tier(brand)
        input_data = pd.DataFrame([[
            brand_id, tier_id, 2024 - model_year, km_driven,
            {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}[seller],
            {'CNG': 0, 'Diesel': 1, 'Electric': 2, 'LPG': 3, 'Petrol': 4}.get(fuel, 4),
            {'Automatic': 0, 'Manual': 1}[trans], mileage, engine, power, seats
        ]], columns=feature_cols)

        fair_price = model.predict(input_data)[0]
        pct_diff = ((asking_price - fair_price) / fair_price) * 100

        st.markdown("<br>", unsafe_allow_html=True)

        c_res1, c_res2 = st.columns([1, 2])
        with c_res1:
            st.markdown(f"""
            <div class="result-card" style="padding: 20px;">
                <div style="font-size: 0.9rem; color: #94a3b8; letter-spacing: 1px;">AI FAIR VALUE</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{format_currency(fair_price)}</div>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
                <div style="font-size: 0.9rem; color: #94a3b8; letter-spacing: 1px;">SELLER ASKING</div>
                <div style="font-size: 1.8rem; font-weight: 700;">{format_currency(asking_price)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_res2:
            st.markdown("**Deal Thermometer**")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=asking_price,
                delta={'reference': fair_price, 'relative': True, "valueformat": ".1%"},
                gauge={
                    'axis': {'range': [fair_price * 0.6, fair_price * 1.4]},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [fair_price * 0.6, fair_price * 0.95], 'color': "#10b981"},
                        {'range': [fair_price * 0.95, fair_price * 1.05], 'color': "#facc15"},
                        {'range': [fair_price * 1.05, fair_price * 1.4], 'color': "#ef4444"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 5}, 'thickness': 0.8, 'value': asking_price}
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20),
                              template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        if pct_diff < -5:
            st.success(f"**Excellent Deal!** This car is priced {abs(pct_diff):.1f}% below market value.")
        elif pct_diff > 5:
            st.error(f"**Overpriced.** The seller is asking {pct_diff:.1f}% more than the fair market value.")
        else:
            st.warning(f"**Fair Deal.** The price is exactly where it should be.")

# ==============================================================================
# VIEW 4: STRATEGIC MARKET SCANNER (PERSONA BASED LOGIC + RESTORED DEFAULTS)
# ==============================================================================
elif view_mode == "Strategic Market Scanner":
    st.markdown('<div class="main-header">Strategic Market Scanner</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
    <strong>Persona-Based Scouting Engine:</strong> 
    Stop browsing blindly. Select your <strong>Driving Persona</strong> below, and the AI will re-weight every car in the market to find your perfect match.
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        # --- SIDEBAR FILTERS ---
        st.markdown('<div class="section-title">1. DEFINE YOUR GOAL</div>', unsafe_allow_html=True)
        with st.container(border=True):
            # THE NEW "USEFUL" FEATURE: BUYING INTENT
            persona = st.radio(
                "What matters most to you?",
                ["⚖️ Balanced Value (Smart Buy)",
                 "📉 Daily Commute (Mileage First)",
                 "🏎️ Performance (Power & Thrills)",
                 "💎 Luxury Status (Badge Value)"],
                index=0
            )

        st.markdown('<div class="section-title">2. SET CONSTRAINTS</div>', unsafe_allow_html=True)
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                # RESTORED DEFAULT: 200000
                min_price = st.number_input("Min Budget (₹)", 0, 50000000, 200000, step=100000)
                # RESTORED DEFAULT: 1000000
                max_price = st.number_input("Max Budget (₹)", 100000, 100000000, 1000000, step=100000)
            with f2:
                # RESTORED DEFAULT: 80000
                max_km = st.number_input("Max Odometer (KM)", 5000, 200000, 80000, step=5000)
            with f3:
                min_year = st.selectbox("Min Registration Year", [2010, 2012, 2014, 2016, 2018, 2020, 2021], index=1)
            with f4:
                tx_type = st.multiselect("Transmission", df_raw['transmission_type'].unique(),
                                         default=['Manual', 'Automatic'])

            st.markdown("<br>", unsafe_allow_html=True)
            run_scan = st.button("🚀 RUN PERSONA SCAN", use_container_width=True)

        if run_scan:
            with st.spinner(f"Analyzing market for '{persona}' profile..."):
                time.sleep(0.5)

                # --- 1. HARD FILTERING ---
                filtered = df_raw[
                    (df_raw['selling_price'] >= min_price) &
                    (df_raw['selling_price'] <= max_price) &
                    (df_raw['year'] >= min_year) &
                    (df_raw['km_driven'] <= max_km) &
                    (df_raw['transmission_type'].isin(tx_type))
                    ].copy()

                if not filtered.empty:
                    # --- 2. DATA NORMALIZATION (PREPARING THE MATH) ---
                    # We normalize all factors between 0 and 1 so we can mix them mathematically

                    # Brand Tier (1=Economy, 3=Luxury) -> Normalized to 0.33, 0.66, 1.0
                    filtered['brand_temp'] = filtered['name'].apply(lambda x: str(x).split(' ')[0])
                    filtered['tier_score'] = filtered['brand_temp'].apply(get_brand_tier)
                    norm_tier = filtered['tier_score'] / 3.0

                    # Year (Newer = 1.0)
                    norm_year = (filtered['year'] - df_raw['year'].min()) / (
                            df_raw['year'].max() - df_raw['year'].min())

                    # Odometer (Lower = 1.0)
                    norm_km = 1 - (filtered['km_driven'] / 200000)
                    norm_km = norm_km.clip(lower=0)

                    # Price Efficiency (Closer to max budget = 1.0)
                    norm_price = filtered['selling_price'] / max_price

                    # Mileage (Higher = 1.0) - Handling potential zeros
                    max_mileage = df_raw['mileage'].max()
                    norm_mileage = filtered['mileage'] / max_mileage

                    # Power (Higher = 1.0)
                    max_power = df_raw['max_power'].max()
                    norm_power = filtered['max_power'] / max_power

                    # --- 3. APPLYING PERSONA WEIGHTS (THE LOGIC) ---

                    if "Daily Commute" in persona:
                        # Logic: Mileage is King (50%), Year matters (30%), Maintenance Tier (Economy is better)
                        # We INVERT Tier score here because Commuters want Economy brands (Tier 1) for cheap parts
                        inv_tier = 1 - (norm_tier - 0.33)  # Rough inversion

                        filtered['score'] = (norm_mileage * 0.50) + (norm_year * 0.30) + (inv_tier * 0.10) + (
                                norm_km * 0.10)
                        strategy_desc = "Prioritizing High Mileage & Low Maintenance"

                    elif "Performance" in persona:
                        # Logic: Power is King (60%), Engine size matters. Fuel efficiency ignored.
                        filtered['score'] = (norm_power * 0.60) + (norm_year * 0.20) + (norm_tier * 0.20)
                        strategy_desc = "Prioritizing BHP, Torque & Engine Capacity"

                    elif "Luxury Status" in persona:
                        # Logic: Brand Badge is King (60%). Year is less important (Old BMW > New Maruti).
                        filtered['score'] = (norm_tier * 0.60) + (norm_price * 0.20) + (norm_year * 0.20)
                        strategy_desc = "Prioritizing Premium Badge & Street Presence"

                    else:  # Balanced Value (Default)
                        # Logic: The "Smart Buy" mix
                        filtered['score'] = (norm_year * 0.40) + (norm_km * 0.30) + (norm_tier * 0.20) + (
                                norm_price * 0.10)
                        strategy_desc = "Prioritizing Reliability, Resale Value & Recency"

                    # Final Scale
                    filtered['score'] = (filtered['score'] * 100).astype(int)
                    filtered = filtered.sort_values('score', ascending=False)
                    top_picks = filtered.head(3)

                    # --- DISPLAY ---
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info(f"🧠 **AI Strategy:** {strategy_desc}")
                    st.markdown(f"**🏆 Top 3 Matches**")

                    c1, c2, c3 = st.columns(3)
                    cols = [c1, c2, c3]

                    for idx, row in enumerate(top_picks.itertuples()):
                        with cols[idx]:
                            badge_color = success_color if row.score > 85 else "#facc15"

                            # Dynamic Info Line based on Persona
                            if "Performance" in persona:
                                extra_info = f"🚀 {row.max_power} BHP"
                            elif "Commute" in persona:
                                extra_info = f"💧 {row.mileage} kmpl"
                            else:
                                extra_info = f"🛣️ {int(row.km_driven / 1000)}k km"

                            st.markdown(f"""
                            <div class="deal-card">
                                <div class="score-badge" style="background: {badge_color}">{row.score}</div>
                                <div class="deal-title">{row.name.title()}</div>
                                <div class="deal-price">{format_currency(row.selling_price)}</div>
                                <div class="deal-specs">
                                    <span>📅 {int(row.year)}</span>
                                    <span>{extra_info}</span>
                                    <span>⛽ {str(row.fuel_type).title()}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-title">FULL RANKING LEDGER</div>', unsafe_allow_html=True)

                    display_df = filtered[
                        ['name', 'selling_price', 'year', 'mileage', 'max_power', 'score']].copy()
                    display_df.columns = ['Vehicle', 'Price', 'Year', 'Mileage', 'Power', 'Match Score']

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        column_config={
                            "Price": st.column_config.ProgressColumn("Price", format="₹%f", min_value=0,
                                                                     max_value=max_price),
                            "Match Score": st.column_config.ProgressColumn("Match Score", min_value=0, max_value=100,
                                                                           format="%d/100")
                        },
                        hide_index=True
                    )

                else:
                    st.warning("No vehicles match your criteria. Try adjusting filters.")

# ==============================================================================
# VIEW 5: MARKET INTELLIGENCE (ORIGINAL - NO CHANGES)
# ==============================================================================
elif view_mode == "Market Intelligence":
    st.markdown('<div class="main-header">Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
    This section explains the "Why" behind the prices. Use these charts to understand which features drive value in the Indian market.
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        tab1, tab2 = st.tabs(["📊 Price Drivers", "🔥 Deep Dive Heatmap"])

        with tab1:
            st.markdown(f"""
            <div class="info-box">
            <strong>How to Read This Data:</strong><br>
            • <strong>Green Bars (Right):</strong> Value Boosters. These features make a car MORE expensive (e.g. More Power, Newer Year).<br>
            • <strong>Red Bars (Left):</strong> Value Reducers. These features make a car CHEAPER (e.g. High Mileage, High Odometer).
            </div>
            """, unsafe_allow_html=True)

            num_df = df_raw.select_dtypes(include=[np.number])
            correlations = num_df.corr()['selling_price'].drop('selling_price').sort_values()

            colors = [danger_color if c < 0 else success_color for c in correlations.values]

            fig = go.Figure(go.Bar(
                x=correlations.values,
                y=[format_labels(i) for i in correlations.index],
                orientation='h',
                marker_color=colors
            ))

            fig.update_layout(
                title="What Drives Car Prices?",
                xaxis_title="Impact on Price (Negative = Cheaper, Positive = Expensive)",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown(f"""
            <div class="info-box">
            <strong>Understanding the Heatmap:</strong><br>
            This grid shows how every feature correlates with every other feature.<br>
            • <strong>Dark Red (1.0):</strong> Strong Positive Relationship (As X goes up, Y goes up).<br>
            • <strong>Dark Blue (-1.0):</strong> Strong Negative Relationship (As X goes up, Y goes down).<br>
            • <strong>White (0):</strong> No relationship.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Feature Correlation Matrix (Expert View)")
            st.caption("This grid shows how every feature relates to the Selling Price and to each other.")

            corr = num_df.corr()
            corr.index = [format_labels(c) for c in corr.index]
            corr.columns = [format_labels(c) for c in corr.columns]

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale='RdBu_r',
                template="plotly_dark",
                labels=dict(color="Correlation")
            )
            st.plotly_chart(fig, use_container_width=True)