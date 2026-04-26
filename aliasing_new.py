import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ====== Streamlit Page Configuration (English, No Chinese Fonts) ======
st.set_page_config(page_title="Signals & Systems Demo", layout="wide")

# ===================== Main Page =====================
st.title("📚 Signals & Systems Teaching Demonstration")
tab_alias, tab_other = st.tabs(["🔧 Sampling Alias Simulator", "📌 Other Modules (Reserved)"])

with tab_alias:
    st.header("Sampling Alias Simulator")
    st.divider()
    sub_tab1, sub_tab2 = st.tabs(["📊 Sampling Aliasing Demo", "🛡 Anti-Aliasing Filter Demo"])

    with sub_tab1:
        st.subheader("Sampling Aliasing Demonstration (Nyquist Sampling Theorem)")
        col1, col2 = st.columns(2)
        with col1:
            f = st.slider(r"Signal Frequency $f\ (\rm Hz)$", 1, 50, 15, key="f_signal")
        with col2:
            fs = st.slider(r"Sampling Frequency $f_{\rm s}\ (\rm Hz)$", 1, 50, 25, key="fs_sample")


        # ====== Sinc Signal Definition (Integrated from First Code) ======
        def sinc_signal(t, f):
            y = np.zeros_like(t)
            nonzero = t != 0
            y[nonzero] = np.sin(2 * np.pi * f * t[nonzero]) / (np.pi * t[nonzero])
            y[~nonzero] = 2 * f  # Limit value at t=0
            return y


        # Time Domain Signal (Symmetric time axis for sinc signal)
        t = np.linspace(-0.5, 0.5, 3000)
        ts = np.arange(-0.5, 0.5, 1 / fs)

        # Original and sampled sinc signal
        x = sinc_signal(t, f)
        xs = sinc_signal(ts, f)

        # Figure 1: Time Domain Waveform
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t, x, label="Original Sinc Signal", lw=1, color="red")
        ax1.vlines(ts, 0, xs, lw=1, colors='green')
        ax1.scatter(ts, xs, color='green', s=10, label="Sampling Points")

        f_alias = 0
        if fs < 2 * f:
            # Calculate aliased frequency
            f_mod = f % fs
            f_alias = fs - f_mod if f_mod > fs / 2 else f_mod
            x_alias = sinc_signal(t, f_alias)

            # Match phase with sampling points
            sign = np.sign(xs[1] / sinc_signal(ts, f_alias)[1])
            x_alias *= sign

            ax1.plot(t, x_alias, '--', lw=1, color='blue', label="Aliased Signal")
            st.warning("⚠ Frequency Aliasing Occurred!")
        else:
            st.success("✅ Satisfies Nyquist Sampling Theorem!")

        ax1.set_title("Time Domain Sinc Signal Waveform")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)


        # FFT Function
        def fft_signal(sig, t):
            N = len(sig)
            dt = t[1] - t[0]
            freq = np.fft.fftfreq(N, dt)
            spectrum = np.abs(np.fft.fft(sig)) / N
            return np.fft.fftshift(freq), np.fft.fftshift(spectrum)


        # Original Signal Spectrum
        freq_x, X = fft_signal(x, t)
        x_sampled = np.zeros_like(t)
        indices = (ts / (t[1] - t[0])).astype(int)
        indices = indices[indices < len(t)]
        x_sampled[indices] = xs[:len(indices)]
        freq_s, Xs = fft_signal(x_sampled, t)

        # Figure 2: Original Signal Spectrum
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(freq_x, X, label="Original Signal Spectrum", color='red')
        ax2.set_xlim(-60, 60)
        ax2.set_title("Original Sinc Signal Spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # Figure 3: Sampled Signal Spectrum
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(freq_s, Xs, label="Sampled Signal Spectrum", color='blue')
        ax3.set_xlim(-60, 60)
        ymax = np.max(Xs)
        ax3.set_ylim(0, ymax * 1.2)

        # Aliasing Region
        if fs < 2 * f:
            f_overlap_start = fs - f
            f_overlap_end = f
            if f_overlap_start < f_overlap_end:
                ax3.fill_betweenx([0, ymax], f_overlap_start, f_overlap_end, color='blue', alpha=0.2,
                                  label="Aliasing Region")
                ax3.fill_betweenx([0, ymax], -f_overlap_end, -f_overlap_start, color='blue', alpha=0.2)


        # Dashed Frequency Lines with labels
        def vline(x, label=None):
            ax3.plot([x, x], [0, ymax], linestyle='--', linewidth=1, color='black')
            if label:
                ax3.text(x, ymax * 1.05, label, ha='center')


        vline(f, r"$+f$")
        vline(-f, r"$-f$")
        vline(fs, r"$+f_s$")
        vline(-fs, r"$-f_s$")

        ax3.set_title("Sampled Signal Spectrum (Periodic Extension)")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Magnitude")
        ax3.legend(loc="upper right")
        ax3.grid(True)
        st.pyplot(fig3)

    with sub_tab2:
        st.subheader("Anti-Aliasing Filter Demonstration")
        st.markdown(
            "**Principle**: Filter out high-frequency components before sampling to avoid aliasing fundamentally.")

        col1, col2, col3 = st.columns(3)
        with col1:
            f_main = st.slider("Main Signal Frequency (Hz)", 1, 20, 10, key="f_main")
        with col2:
            f_noise = st.slider("High-Frequency Noise Frequency (Hz)", 45, 100, 60, key="f_noise")
        with col3:
            fs = st.slider("Sampling Frequency fs (Hz)", 75, 100, 80, key="fs_filter")

        fc = st.slider("Filter Cut-off Frequency (Hz)", 25, 40, 30, key="fc")

        # Generate Signals
        t = np.linspace(0, 1, 1000)
        x_mix = np.sin(2 * np.pi * f_main * t) + 0.3 * np.sin(2 * np.pi * f_noise * t)
        x_filtered = np.sin(2 * np.pi * f_main * t)

        ts = np.arange(0, 1, 1 / fs)
        xs_mix = np.sin(2 * np.pi * f_main * ts) + 0.3 * np.sin(2 * np.pi * f_noise * ts)
        xs_filtered = np.sin(2 * np.pi * f_main * ts)

        # Figure 4: Anti-Aliasing Filter Comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(t, x_mix, color="red", label="Mixed Signal (with High-Frequency Noise)")
        ax1.scatter(ts, xs_mix, color="green", label="Sampling Points")
        ax1.set_title("Without Anti-Aliasing Filter (Prone to Aliasing)")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(t, x_filtered, color="blue", label="Filtered Signal (No High-Frequency Noise)")
        ax2.scatter(ts, xs_filtered, color="green", label="Sampling Points")
        ax2.set_title("With Anti-Aliasing Filter (Eliminate Aliasing)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)

        # Prompt Messages
        if f_noise > fs / 2:
            st.success("✅ Anti-aliasing filter removed high-frequency noise, completely avoiding aliasing!")
        else:
            st.info("ℹ️ Noise frequency is below Nyquist frequency, no obvious aliasing.")

with tab_other:
    st.info("ℹ️ More signals and systems teaching modules can be extended here.")