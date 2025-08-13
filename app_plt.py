# app_tise_merge.py
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TISE Tutor — Harmonic Oscillator", layout="centered")

# ---------------- Session state init ----------------
if "psi_library" not in st.session_state:
    st.session_state["psi_library"] = []  # list of dicts: {label, E, L, x, psi}

# ---------------- Heavy compute (cached + gated by button) ----------------
@st.cache_data(show_spinner=False)
def solve_once(E_input: float, L: float, N_fixed: int, match_tol: float):
    """Compute grid, potential, and wavefunction (eigen if near-right; else Numerov shooting)."""
    x = np.linspace(-L, L, int(N_fixed))
    h = x[1] - x[0]
    V = 0.5 * x**2

    # finite-diff eigensolve
    main = np.full(x.size, -2.0)
    off  = np.full(x.size - 1, 1.0)
    D2 = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (h**2)
    H  = -0.5 * D2 + np.diag(V)
    evals, evecs = np.linalg.eigh(H)
    k_idx = int(np.argmin(np.abs(evals - E_input)))
    E_best = float(evals[k_idx])
    psi_best = evecs[:, k_idx]
    psi_best /= np.sqrt(np.trapezoid(psi_best**2, x))

    # if not near an eigenvalue, shoot with Numerov using left boundary from psi_best
    near_right = abs(E_input - E_best) < match_tol
    if near_right:
        psi_used = psi_best
    else:
        kx = 2.0 * (V - E_input)
        psi = np.zeros_like(x)
        psi0, psi1, psi2 = psi_best[0], psi_best[1], psi_best[2]
        psi_prime0 = (-3*psi0 + 4*psi1 - psi2) / (2*h)
        psi[0] = psi0
        psi[1] = psi0 + h*psi_prime0 + 0.5*kx[0]*(h**2)*psi0
        for i in range(1, x.size-1):
            denom = 1 + (h**2) * kx[i+1] / 12.0
            term1 = 2 * (1 - 5 * (h**2) * kx[i] / 12.0) * psi[i]
            term2 = (1 + (h**2) * kx[i-1] / 12.0) * psi[i-1]
            psi[i+1] = (term1 - term2) / denom
        norm = np.trapezoid(psi**2, x)
        psi_used = psi / np.sqrt(norm) if norm > 0 and np.isfinite(norm) else psi

    return x, V, psi_used, E_best, near_right

# ---------------- Controls ----------------
with st.sidebar:
    st.subheader("Controls")
    E = st.number_input("Energy E", value=1.5, step=0.1, format="%.6f")
    L = st.slider("Domain half-width L", 4.0, 12.0, 8.0, 0.5)
    #tol = st.number_input("|E−E_k| near-right tol", value=5e-3, format="%.1e")
    calc = st.button("Calculate", type="primary")

# Fixed grid points (sidebar option removed)
N_FIXED = 401
tol = 5e-3
# compute ONCE (on calc or first run)
if calc or ("solution" not in st.session_state):
    x, V, psi, E_best, near_right = solve_once(E, L, N_FIXED, tol)
    st.session_state["solution"] = dict(x=x, V=V, psi=psi, E=E, E_best=E_best, near=near_right, L=L)

# Pull solution
sol = st.session_state["solution"]
x, V, psi, E, E_best, near_right, L = (sol[k] for k in ["x", "V", "psi", "E", "E_best", "near", "L"])

# ---------------- Sidebar: Wavefunction library (max 2) ----------------
show_indices = []

with st.sidebar:
    st.divider()
    st.subheader("Wavefunction library (max 2)")

    # name to store current ψ
    default_label = f"ψ(E={E:.3f})"
    new_label = st.text_input("Label to store", value=default_label)
    store_btn = st.button("➕ Store current ψ", help="Save the current computed wavefunction")

    if store_btn:
        if len(st.session_state["psi_library"]) >= 2:
            st.warning("Maximum of 2 stored wavefunctions. Remove one before adding another.")
        else:
            entry = {
                "label": (new_label.strip() or default_label),
                "E": float(E),
                "L": float(L),
                "x": np.array(x, dtype=float),
                "psi": np.array(psi, dtype=float),
            }
            st.session_state["psi_library"].append(entry)
            st.success(f"Stored: {entry['label']}")

    # List stored wavefunctions with show / download / remove
    if st.session_state["psi_library"]:
        st.caption("Toggle to overlay on the plot:")
        for i, item in enumerate(st.session_state["psi_library"]):
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                on = st.checkbox(f"{item['label']} (E={item['E']:.3f})", key=f"lib_show_{i}")
                if on:
                    show_indices.append(i)
            with c2:
                csv = "x,psi\n" + "\n".join(f"{xx},{pp}" for xx, pp in zip(item["x"], item["psi"]))
                st.download_button(
                    "CSV",
                    data=csv.encode(),
                    file_name=f"{item['label'].replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"lib_dl_{i}",
                    help="Download this ψ as CSV",
                )
            with c3:
                if st.button("❌", key=f"lib_rm_{i}", help="Remove from library"):
                    st.session_state["psi_library"].pop(i)
                    st.rerun()
        st.button("🗑️ Clear library", on_click=lambda: st.session_state.update(psi_library=[]))
    else:
        st.caption("No stored wavefunctions yet.")

    # ---------- Merge two stored ψ into one ----------
    st.divider()
    st.subheader("Merge two stored ψ")
    if len(st.session_state["psi_library"]) == 2:
        merge_method = st.selectbox("Merge method", ["Average (normalized)", "Sum (normalized)"])
        merge_label = st.text_input(
            "Merged label",
            value=f"Merge: {st.session_state['psi_library'][0]['label']} + {st.session_state['psi_library'][1]['label']}"
        )
        merge_show = st.checkbox("Show merged ψ immediately", value=True)
        merge_btn = st.button("🔀 Merge wavefunctions")
        
        if merge_btn:
            wf1 = st.session_state["psi_library"][0]
            wf2 = st.session_state["psi_library"][1]
            if not np.allclose(wf1["x"], wf2["x"]):
                st.error("Cannot merge — x grids differ.")
            else:
                if merge_method.startswith("Average"):
                     psi_merged = 0.5 * (wf1["psi"] + wf2["psi"])
                else:
                     psi_merged = wf1["psi"] + wf2["psi"]
            # normalize
            norm = np.sqrt(np.trapezoid(psi_merged**2, wf1["x"]))
            if norm > 0 and np.isfinite(norm):
                psi_merged /= norm

            # ----- Estimate new energy -----
            xg = wf1["x"]
            Lg = wf1["L"]
            Vg = 0.5 * xg**2
            h = xg[1] - xg[0]
            d2psi = np.zeros_like(psi_merged)
            d2psi[1:-1] = (psi_merged[2:] - 2*psi_merged[1:-1] + psi_merged[:-2]) / h**2
            kinetic = -0.5 * np.trapezoid(psi_merged * d2psi, xg)
            potential = np.trapezoid(Vg * psi_merged**2, xg)
            E_est = kinetic + potential

            entry = {
                "label": (merge_label.strip() or "Merged ψ"),
                "E": float(E_est),
                "L": Lg,
                "x": xg.copy(),
                "psi": psi_merged,
            }
            # replace the two originals with the single merged ψ
            st.session_state["psi_library"] = [entry]
            st.success(f"Merged ψ saved: {entry['label']}, estimated energy: {E_est:.6f}")
            if merge_show:
                show_indices = [0]
    else:
       st.caption("Need exactly 2 stored ψ to merge.")

# --------- Precompute curvature, T(x), and rearranged LHS for hover ---------
h = x[1] - x[0]
psi_pp = np.zeros_like(psi)
psi_pp[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / (h**2)
psi_pp[0] = psi_pp[1]
psi_pp[-1] = psi_pp[-2]
T_local = E - V

# constants in our units: m=1, ħ=1 ⇒ c = ħ²/2 = 1/2 ⇒ (c/m)=1/2
c_over_m = 0.5
lhs_term = np.full_like(psi, np.nan)
mask = np.abs(psi) > 1e-12
lhs_term[mask] = -c_over_m * (psi_pp[mask] / psi[mask]) + V[mask]

# ---------------- Equations (original + rearranged) ----------------
st.title("Introduction of wavefunction")
st.markdown("""
Before the discovery of the Schrödinger equation in 1926, numerous models were proposed to describe the structure of the atom, including J.J. Thomson’s “plum pudding” model (1904) and Ernest Rutherford’s nuclear model (1911). Rutherford’s gold foil experiment in 1909 demonstrated that atoms consist of a small, dense, positively charged nucleus surrounded by electrons.

The key theoretical challenge was to explain how electrons could remain in stable orbits around the nucleus without spiraling inward due to electromagnetic attraction. In 1913, Niels Bohr proposed a quantized orbit model in which electrons could occupy only specific energy levels, preventing them from collapsing into the nucleus. While Bohr’s model successfully explained the hydrogen spectrum, it could not fully account for more complex atoms.

A breakthrough came with the recognition of the wave–particle duality of matter. In 1924, Louis de Broglie proposed that electrons possess wave-like properties, suggesting that their allowed orbits correspond to standing waves. In 1927, the Davisson–Germer experiment confirmed electron diffraction, providing direct evidence of their wave nature. This wave–particle duality laid the foundation for the quantum mechanical description of electrons in atoms, as formulated by Schrödinger in 1926. Could this helps with the answer to the question?
""")

st.image('electrostatic_stability_cartoon.png', width=600)
st.markdown("**Electrostatic (Coulomb) potential:**")
st.latex(r"V(r) = -\frac{k_e \, q_1 q_2}{r}")
st.caption(
    r"where $k_e$ is Coulomb's constant, $q_1$ and $q_2$ are the charges, "
    r"and $r$ is the distance between them."
)

st.markdown("""
**According to Earnshaw’s theorem (1839), a 1/r potential has no stable stationary point**—meaning that in a system governed solely by inverse-square forces, such as electrostatics or gravity, no arrangement of particles can be in stable equilibrium at rest.  
Kinetic energy is therefore essential for stability. To illustrate this, let us consider the analogous case of the gravitational potential.
""")

st.markdown("**Newtonian gravitational potential:**")
st.latex(r"V(r) = -\frac{G m_1 m_2}{r}")
st.caption(
    r"where $G$ is the gravitational constant, $m_1$ and $m_2$ are masses, "
    r"and $r$ is the distance between their centers."
)
st.image("gravity_orbit.gif")

st.markdown("""
**How about the electron–nucleus system?**  
Remember, the electron is not only a particle but also a wave! Schrödinger equation was proposed.
""")

st.markdown("**Schrödinger (time‑independent) — original form (1D case):**")
st.latex(r"-\frac{\hbar^2}{2m}\,\psi''(x) + V(x)\,\psi(x) = E\,\psi(x)")

st.markdown(r"""
Let’s take a closer look at the equation.  
Here, $E$ is constant, $V$ is a function of position, and $\psi$ is also a function of position.  

Schrödinger introduced an operation on the wavefunction to calculate the kinetic energy — this appears as the first term in the equation.  
Once the wavefunction is known, we can determine the total energy of the system.  
Because the electron is also a particle, this total energy must be constant; it cannot have different values at the same instant in time.  

The kinetic energy term is directly related to the ratio of the wavefunction’s curvature to the wavefunction itself.  
With this in mind, let us rearrange the equation.
""")

st.markdown("**Rearranged:**")
st.latex(r"\text{Kinetic energy:}\quad -\frac{c}{m}\,\frac{\psi''(x)}{\psi(x)} = E - V(x)")
st.caption(r"where $c=\frac{\hbar^2}{2}$ and (in this app) $m=1,\ \hbar=1$ so $c/m=\tfrac12$.")

st.markdown(r"""
Let’s start with a **harmonic potential** as our model.  
It’s straightforward to implement in 1D and has an energy minimum at $r=0$.  
Imagine placing a nucleus at the center of the well.

In the **classical case**, total energy is conserved.  
Since kinetic energy is always positive $\tfrac{1}{2} m v^2$,  
the highest position the particle can reach is where its speed becomes zero.
""")

st.image("harmonic_particle_on_potential.gif")

st.markdown("""
Let’s solve the Schrödinger equation for the same potential.  
In the sidebar, guess an energy value and click **Calculate**.  
You can store wavefunctions for later comparison.

You’ll soon discover that the energy is **quantized** — only values like **0.5, 1.5, 2.5**, etc., are allowed.  
Wavefunctions with other energies will diverge.

Hover over the wavefunction plot and observe:

1. **Inside the two vertical bars:**  
   The electron behaves similarly to a classical particle — the kinetic energy (total energy minus potential energy) is always positive.  
   A useful trick: in this region, the wavefunction always curves back toward the baseline.

2. **At the bars:**  
   The kinetic energy is zero, which means the wavefunction curvature is also zero.

3. **Outside the bars:**  
   The kinetic energy is **negative** — yes, that’s allowed in quantum mechanics!  
   At the far ends, both the curvature and the wavefunction value approach zero,  
   resulting in a large but finite difference between the potential and total energies.
""")

# ---------------- Figure ----------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=V, name="V(x)=½x²", mode="lines"))
fig.add_hline(y=E, line_dash="dash", annotation_text=f"E={E:.4f}")

# main ψ with rich hover (only ψ, ψ″, large font)
fig.add_trace(
    go.Scatter(
        x=x,
        y=psi,
        name="ψ(x)",
        yaxis="y2",
        mode="lines",
        customdata=np.stack([psi, psi_pp, T_local], axis=-1),
        hovertemplate=(
            "<b>ψ(x)</b> = %{customdata[0]:+.3e}<br>"
            "<b>ψ″(x)</b> ≈ %{customdata[1]:+.3e}<br>"
            "<b>K.E.</b> = %{customdata[2]:+.3e}<br>"
            "<extra></extra>"  # removes the trace name from hover box
        ),
    )
)

fig.update_layout(
    hoverlabel=dict(
        font_size=18,
        font_family="Arial"
    )
)

# overlay any selected stored ψ (thin, dotted)
for i in show_indices:
    item = st.session_state["psi_library"][i]
    fig.add_trace(
        go.Scatter(
            x=item["x"],
            y=item["psi"],
            name=item["label"],
            yaxis="y2",
            mode="lines",
            line=dict(width=1, dash="dot"),
            opacity=0.9,
        )
    )

# turning points if E≥0
if E >= 0:
    xL_true, xR_true = -np.sqrt(2*E), np.sqrt(2*E)
    fig.add_vline(x=xL_true, line=dict(color="red", width=3))
    fig.add_vline(x=xR_true, line=dict(color="red", width=3))

fig.update_layout(
    height=560,
    hovermode="x unified",
    hoverlabel=dict(
        font_size=18,        # larger hover font
        font_family="Arial"  # or any font you like
    ),
    xaxis=dict(title="x", range=[-L, L]),
    yaxis=dict(title="Energy"),
    yaxis2=dict(title="ψ", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Nearest grid eigenvalue: `E_k = {E_best:.6f}`; "
    f"Near-right? `{'YES' if near_right else 'NO'}` (|E−E_k| = {abs(E - E_best):.3e}). "
    "Turning points shown at x = ±√(2E). Hover the ψ curve to see ψ and ψ″."
)

st.markdown("""
**Thinking prompts**  
1. If we define a *nodal point* as the position where the wavefunction crosses the baseline, the energy tends to increase with the number of nodal points. Try comparing **E = 0.5** and **E = 2.5** — why might this be the case?  
2. You may have heard of the term *tunneling*. Can you think of an explanation for this phenomenon?
3. **Most important:** Do we now have an answer to the original question — why doesn’t the wavefunction simply localize at the lowest potential point?
""")

# Q1
with st.expander("Q1: Show answer"):
    st.markdown(r"""
More nodes mean higher curvature in the wavefunction, and thus higher kinetic energy.  
From the Schrödinger equation, the kinetic term is:

$$
T \;=\; -\frac{\hbar^2}{2m} \frac{\psi''(x)}{\psi(x)}
$$

A greater number of nodes corresponds to a larger effective wave number \(k\), giving:

$$
\langle T \rangle \sim \frac{\hbar^2 k^2}{2m}
$$

Thus, more nodes → larger \(k\) → higher total energy.
""")

# Q2
with st.expander("Q2: Show answer"):
    st.markdown(r"""
At first glance, what sets a quantum system apart from a classical one is the possibility of having **negative kinetic energy** in certain regions — this allows an electron to *escape* the confinement of a potential well.  

In regions where \( E < V(x) \), the Schrödinger equation has solutions that **decay exponentially**:  
$$
\psi(x) \propto e^{-\kappa x}, 
\quad 
\kappa = \frac{\sqrt{2m \, (V - E)}}{\hbar}.
$$

This means the wavefunction has a **finite amplitude** beyond the classical turning point,  
implying a nonzero probability of finding the particle outside the classically allowed region.  
This phenomenon is known as **quantum tunneling**.
""")

# Q3
with st.expander("Q3 (Most important): Show answer"):
    st.markdown(r"""
The wavefunction cannot collapse into a single point at the potential minimum because:

1. **Kinetic energy penalty (cause):**  
   From the Schrödinger equation, strong localization forces large curvature $|\psi''(x)|$,  
   which directly increases  
   $$
   T = -\frac{\hbar^2}{2m} \frac{\psi''(x)}{\psi(x)}.
   $$

2. **Uncertainty principle (consequence):**  
   In momentum space, this appears as  
   $$
   \Delta x \, \Delta p \gtrsim \frac{\hbar}{2},
   $$  
   where small $\Delta x$ means large $\Delta p$ and  
   $$
   T \sim \frac{(\Delta p)^2}{2m}.
   $$  
   This is not a separate cause, but another way to see the kinetic energy cost. As confinement becomes stronger (smaller $\Delta x$), the effective potential well becomes steeper,  
   which increases the spacing between quantized energy levels.

**Result:**  
The ground state spreads out to minimize total energy, balancing potential energy gain near the minimum against the kinetic energy cost of localization.
""")

st.markdown("""
**Let us bring two wells together**  
One well is deeper than the other.  
Now, check whether the conclusions from our previous discussion still hold:

1. **Energy gaps in the deeper well are wider** — stronger confinement leads to larger spacing between quantized levels.  
2. **Tunneling between wells** — wavefunctions can extend into the neighboring well, allowing states to have nonzero amplitude in both wells.
""")
st.image("lowest_eigenstates_0.png")
st.image("lowest_eigenstates_1.png")