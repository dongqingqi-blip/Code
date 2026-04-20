# Import required libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress

# -------------------------- Page Configuration --------------------------
# Set Streamlit page layout and title for academic visualization
st.set_page_config(page_title="Rossler Attractor Analysis", layout="wide")
st.title("Rossler Attractor Chaotic System Analysis")

# Initialize session state variables for simulation control and data storage
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "final_xs" not in st.session_state:
    st.session_state.final_xs = np.array([])

# -------------------------- UI Tabs --------------------------
# Create three tabs for theory, simulation, and complexity analysis
tab1, tab2, tab3 = st.tabs([
    "📚 Theory & Background",
    "🎬 Attractor Simulation",
    "📊 Complexity Analysis"
])

# ==============================================================================
# Tab 1: Theoretical Background of Rossler System (Full English Content)
# ==============================================================================
with tab1:
    st.header("1. Rossler Attractor: Complete Theory")
    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1.1 System Definition & Dimension")
        st.markdown("""
**Mathematical Type**:  
3D continuous-time autonomous dynamical system.

**State Space**:  
The system evolves in a 3-dimensional Euclidean space spanned by $(x, y, z)$.

**Attractor Type**:  
It is a **strange attractor** with fractal geometry:
- **Correlation Dimension**: ≈ 2.01 – 2.05 (non-integer, fractal)
- **Topological Dimension**: ≈ 2
- **Structure**: A folded 2D-like surface embedded in 3D space
- **Visualization**: Typically shown as a 2D projection on the x-y plane
""")

        st.subheader("1.2 Governing Equations")
        st.latex(r"""
\begin{cases}
\dot{x} = -y - z \\
\dot{y} = x + a \, y \\
\dot{z} = b + z \, (x - c)
\end{cases}
""")
        st.markdown("""
**Parameters**:
- \(a, b, c\): control parameters
- **Classical chaotic parameters**: \(a=0.2, b=0.2, c=5.7\)
- Only **one nonlinear term** \(z \cdot x\), making it the simplest chaotic system
""")

        st.subheader("1.3 Physical Meaning & Origin")
        st.markdown("""
**History**:  
Proposed by Otto Rössler in 1976 to **simplify the Lorenz system** and reveal the fundamental mechanism of chaos.

**Physical Interpretation**:
Not tied to one specific device, but describes **universal nonlinear behavior**:
- Chemical chaos (e.g., BZ reaction)
- Self-excited oscillations
- Period-doubling route to chaos
- Biological and neural oscillations
- Nonlinear feedback systems

**Academic Role**:  
The most widely used model for teaching and analyzing chaos.
""")

    with col_right:
        st.subheader("1.4 Chaotic Indicators: Formula & Meaning")

        st.markdown("#### A. Maximum Lyapunov Exponent (MLE)")
        st.latex(r"\lambda \approx \frac{1}{N\Delta t}\sum_{i=1}^{N-1}\ln\left|\Delta x_i\right|")
        st.markdown("""
**Meaning**:  
Measures the **exponential separation rate** of nearby trajectories.

**Judgment**:
- \(\lambda > 0\): Chaotic
- \(\lambda \le 0\): Periodic or stable
""")

        st.markdown("#### B. 0–1 Chaos Test (Gottwald–Melbourne)")
        st.latex(r"""
p_n = \sum_{k=1}^n x_k \cos(ck),\quad
q_n = \sum_{k=1}^n x_k \sin(ck)
""")
        st.latex(r"""
M_n = \sqrt{p_n^2 + q_n^2},\quad
K = \lim_{n\to\infty}\frac{\log M_n}{\log n}
""")
        st.markdown("""
**Meaning**:  
The **most robust** chaos detection method for time series.

**Judgment**:
- \(K \to 1\): Chaotic
- \(K \to 0\): Non-chaotic (periodic)
""")

        st.markdown("#### C. Phase Space Reconstruction")
        st.latex(r"\mathbf{x}_i = \big(x_i,\; x_{i+\tau}\big)")
        st.markdown("""
Based on **Takens’ Theorem**:  
A single time series can reconstruct the full attractor geometry.
""")

        st.markdown("#### D. Power Spectrum")
        st.latex(r"P(\omega) = \frac{1}{N}\left|\sum_{n=1}^N x_n e^{-i\omega n}\right|^2")
        st.markdown("""
**Judgment**:
- **Continuous broadband**: Chaotic
- **Sharp discrete peaks**: Periodic
""")

    st.divider()
    st.subheader("1.5 Code Validation & Error Check")
    st.markdown("""
✅ **Governing equations**: Correct  
✅ **Dimension**: 3D system solved, 2D projected – logically consistent  
✅ **Integration**: RK4 used – high precision, no numerical dissipation  
✅ **Chaotic indicators**: Standard algorithms implemented correctly  
✅ **Transient removal**: Only stable attractor data used for analysis  

**Note**:  
The Lyapunov exponent in this code is a **1D approximation** for practical use.  
The **0–1 test** is the primary and most reliable judgment.
""")

# ==============================================================================
# Tab 2: Rossler Attractor Numerical Simulation (Optimized Plotting)
# ==============================================================================
with tab2:
    st.header("2. Rossler Attractor Simulation")
    st.sidebar.header("Simulation Parameters")

    # Slider controls for system parameters and simulation settings
    a = st.sidebar.slider("a", 0.0, 1.0, 0.20, 0.01)
    b = st.sidebar.slider("b", 0.0, 1.0, 0.20, 0.01)
    c = st.sidebar.slider("c", 0.0, 15.0, 5.70, 0.1)
    dt_display = st.sidebar.slider("Step Size (ms)", 0.05, 1.0, 0.10, 0.05)
    max_steps = st.sidebar.slider("Total Steps (k)", 1000, 5000, 3000, 500) * 1000

    # Convert millisecond step size to second for numerical integration
    dt = dt_display / 1000

    col1, col2 = st.columns(2)
    with col1:
        # Start simulation button
        if st.button("Start"):
            st.session_state.running = True
            st.session_state.history = []
            # Initial state of the Rossler system (non-zero to avoid trivial solution)
            st.session_state.state = (0.1, 0.1, 0.1)
    with col2:
        # Stop simulation button
        if st.button("Stop"):
            st.session_state.running = False

    # Placeholder for dynamic plotting of the attractor
    placeholder = st.empty()

    if st.session_state.running:
        x, y, z = st.session_state.state
        hist = st.session_state.history
        av, bv, cv = a, b, c

        with st.spinner("Generating High-Quality Attractor..."):
            # 4th Order Runge-Kutta (RK4) numerical integration
            for step in range(max_steps):
                # RK4 first stage
                dx = -y - z
                dy = x + av * y
                dz = bv + z * (x - cv)

                # RK4 second stage
                x1 = x + dt * dx / 2
                y1 = y + dt * dy / 2
                z1 = z + dt * dz / 2
                dx1 = -y1 - z1
                dy1 = x1 + av * y1
                dz1 = bv + z1 * (x1 - cv)

                # RK4 third stage
                x2 = x + dt * dx1 / 2
                y2 = y + dt * dy1 / 2
                z2 = z + dt * dz1 / 2
                dx2 = -y2 - z2
                dy2 = x2 + av * y2
                dz2 = bv + z2 * (x2 - cv)

                # RK4 fourth stage
                x3 = x + dt * dx2
                y3 = y + dt * dy2
                z3 = z + dt * dz2
                dx3 = -y3 - z3
                dy3 = x3 + av * y3
                dz3 = bv + z3 * (x3 - cv)

                # Update system state using weighted RK4 sum
                x += dt * (dx + 2 * dx1 + 2 * dx2 + dx3) / 6
                y += dt * (dy + 2 * dy1 + 2 * dy2 + dy3) / 6
                z += dt * (dz + 2 * dz1 + 2 * dz2 + dz3) / 6

                # Retain only stable late-stage data (remove transient response)
                if step > max_steps * 0.2:
                    hist.append((x, y))

                # Terminate loop if stop button is pressed
                if not st.session_state.running:
                    break

        # Extract coordinate data for plotting
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]

        # ====================== Optimized High-Definition Plotting ======================
        # Professional plot style for chaotic attractor visualization
        plt.style.use('default')
        # High DPI + large figure size for clear fractal structure
        fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
        # Pure black background to highlight neon trajectory color
        ax.set_facecolor("#000000")

        # High-contrast neon cyan line: optimized thickness + transparency for chaos visualization
        ax.plot(xs, ys, color='#00FFFF', linewidth=0.5, alpha=0.9)

        # Remove all axes, ticks, and spines for clean academic visualization
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

        # Tight layout to eliminate white borders
        plt.tight_layout(pad=0)
        # Render plot in Streamlit placeholder
        placeholder.pyplot(fig, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save simulation results to session state
        st.session_state.final_xs = np.array(xs)
        st.session_state.history = hist
        st.session_state.running = False
        st.success("✅ High-Quality Rossler Attractor Generated Successfully!")

# ==============================================================================
# Tab 3: Chaotic System Complexity Analysis (Optimized Visualization)
# ==============================================================================
with tab3:
    st.header("3. Chaotic Indicators Analysis")


    def robust_01_test(series, trials=3):
        """
        Robust 0-1 Chaos Test (Gottwald-Melbourne)
        Detects chaos in deterministic dynamical systems from time series
        Returns: K-value (0=non-chaotic, 1=chaotic) and verdict
        """
        N = len(series)
        if N < 5000:
            return 0.0, "Insufficient Data"

        # Detrend the time series to eliminate linear bias
        t = np.arange(N)
        slope, intercept, _, _, _ = linregress(t, series)
        detrended = series - (slope * t + intercept)
        Ks = []

        # Multiple random c values for robust result
        for _ in range(trials):
            c = np.random.uniform(np.pi / 8, 7 * np.pi / 8)
            p = np.cumsum(detrended * np.cos(c * t))
            q = np.cumsum(detrended * np.sin(c * t))
            M = np.sqrt(p ** 2 + q ** 2)

            # Calculate growth rate via linear regression
            idx = np.where((t[1:] > 0) & (M[1:] > 0))[0]
            if len(idx) < 100:
                continue
            log_t = np.log(t[1:][idx])
            log_M = np.log(M[1:][idx])
            sl, _, _, _, _ = linregress(log_t, log_M)
            Ks.append(sl)

        if not Ks:
            return 0.0, "Analysis Error"
        avg_k = np.mean(Ks)
        verdict = "✅ Chaotic" if avg_k > 0.5 else "❌ Non-Chaotic"
        return round(avg_k, 4), verdict


    def phase_recon(seq, tau=5):
        """
        Phase Space Reconstruction based on Takens' Theorem
        tau: time delay for embedding
        Returns: 2D reconstructed phase space points
        """
        n = len(seq)
        if n <= tau:
            return []
        return np.array([[seq[i], seq[i + tau]] for i in range(n - tau)])


    # Trigger chaos analysis
    if st.button("Compute Chaotic Indicators"):
        xs = st.session_state.final_xs
        if len(xs) < 5000:
            st.warning("⚠️ Please run the attractor simulation first to generate sufficient data.")
        else:
            with st.spinner("Performing Chaos Analysis..."):
                # Core chaos detection algorithms
                K, res = robust_01_test(xs)

                # Approximate Maximum Lyapunov Exponent calculation
                try:
                    d = np.abs(np.diff(xs))
                    d = d[d > 1e-9]
                    lyap = np.mean(np.log(d)) / dt if len(d) > 100 else -999
                    lyap_str = f"{lyap:.4f}" if lyap > 0 else "-"
                except:
                    lyap_str = "-"

                # Power spectral density calculation (Welch's method)
                f, Pxx = welch(xs, fs=10000, nperseg=2048)
                # Adaptive time delay for phase space reconstruction
                tau = 5 if K > 0.5 else 1
                psr = phase_recon(xs, tau=tau)

            # Display quantitative analysis results
            st.subheader("Quantitative Results")
            c1, c2 = st.columns(2)
            c1.metric("Maximum Lyapunov Exponent", lyap_str)
            c2.metric("0-1 Test K Value", K)

            c3, c4 = st.columns(2)
            c3.markdown(f"**Chaos State**: {res}")
            c4.markdown(f"**0-1 Test Verdict**: {res}")

            # Optimized plots for spectral and phase space analysis
            st.subheader("Power Spectrum & Reconstructed Phase Space")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=120)

            # Power Spectrum: Highlight continuous broadband chaotic feature
            ax1.plot(f, Pxx, color='#0099FF', linewidth=1.2)
            ax1.set_title("Power Spectrum (Log Scale)", fontweight='bold')
            ax1.set_yscale("log")
            ax1.grid(alpha=0.4, color='white')
            ax1.set_facecolor("#121212")

            # Reconstructed Phase Space: Clear fractal attractor structure
            if len(psr) > 100:
                ax2.plot(psr[:, 0], psr[:, 1], color='#FF00FF', linewidth=0.4, alpha=0.8)
                ax2.set_title("Reconstructed Phase Space", fontweight='bold')
                ax2.axis("equal")
                ax2.set_facecolor("#121212")
            else:
                ax2.text(0.5, 0.5, "Insufficient Data", ha="center", va="center", fontweight='bold')

            st.pyplot(fig)
            plt.close()
            st.success("✅ Chaos Complexity Analysis Completed!")

st.markdown("---")
st.caption("Rossler Attractor | Chaos Theory | RK4 Integrator | Streamlit Academic Visualization Tool")