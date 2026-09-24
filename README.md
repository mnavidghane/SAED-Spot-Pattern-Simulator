# SAED Spot Pattern Simulator

Interactive Python simulator of **Selected Area Electron Diffraction (SAED) spot patterns**
for **all 14 Bravais lattices**, arbitrary zone axes, and arbitrary TEM accelerating voltages.

The program builds the **reciprocal lattice** from the direct-cell parameters, applies the
**zone law** and **centering extinction rules**, computes **Ewald-sphere excitation errors**
and **thickness-dependent intensities**, and renders an **indexed, book-style diffraction
pattern** (black background, color-coded Laue zones, labeled reference spots) together with
a complete text summary.

Written as a **teaching and practical aid for TEM diffraction analysis**: generate the
pattern you expect before (or after) the microscope session and compare it directly with
the experimental one.

---

## 1. Executive Summary

| Item | Description |
|---|---|
| Language / dependencies | Python 3.8+, NumPy, Matplotlib |
| Interface | Interactive console (step-by-step prompts) |
| Lattices | All 14 Bravais types (incl. both trigonal settings: hexagonal and rhombohedral) |
| Zone axis | Arbitrary integer [u v w] |
| Voltage | Arbitrary kV → relativistic electron wavelength |
| Laue zones | ZOLZ / FOLZ / SOLZ / HOLZ (selectable), zone-law filtering, per-zone damping |
| Extinctions | Lattice centering: P, I, F, C, R |
| Intensity model | Geometric: sinc²(s·t) × zone damping (see §10 for scope) |
| Reference indexing | Robust two-reference-spot selection |
| Outputs | Book-like indexed figure + full `SAED SUMMARY` text report |

---

## 2. What the program computes (processing pipeline)

1. **Direct basis** from (a, b, c, α, β, γ) — valid for all crystal systems including
   triclinic; degenerate cells (γ → 0°/180°, non-positive volume factor) are rejected.
2. **Reciprocal basis** — a* = 2π (b×c)/V, b* = 2π (c×a)/V, c* = 2π (a×b)/V.
   The 2π convention is used consistently throughout (k = 2π/λ).
3. **Relativistic electron wavelength**:
   `λ(Å) = 12.3986 / sqrt( V · (1 + 0.97845e-6 · V) )`, V in volts.
4. **Zone law** — n = h·u + k·v + l·w; keep `|n| ≤ Nmax`; label: 0 → ZOLZ, 1 → FOLZ,
   2 → SOLZ, ≥ 3 → HOLZ. `Nmax` is set automatically from the zone selection
   (ZOLZ→0, FOLZ→1, SOLZ→2; mixed selections ask explicitly).
5. **Centering extinction rules** (systematic absences) — see §2.2.
6. **Reciprocal vectors** g(hkl) = h·a* + k·b* + l·c*, with d = 2π/|g|.
7. **Projection onto the diffraction plane** perpendicular to the zone axis;
   the pattern is rotated so that reference spot 1 lies along +x.
8. **Excitation error** (if Ewald mode is on):
   `s ≈ g∥ + |g|² / (2k)`, k = 2π/λ — the small-angle deviation of the reciprocal point
   from the Ewald sphere. If Ewald mode is off, s = 0 (ideal "flat" book pattern).
9. **Intensity**: I_raw = sinc²(s·t) × zone damping (t in Å).
10. **Normalization & display**: I/I_max → contrast power γ → cutoff I_min.
11. **Reference-spot selection** for indexing → figure + `SAED SUMMARY`.

### 2.1 Zone-law / Laue-zone bookkeeping

| |n| = \|hu+kv+lw\| | Label | Meaning |
|---|---|---|
| 0 | ZOLZ | Zero-Order Laue Zone (the familiar central net) |
| 1 | FOLZ | First-Order Laue Zone |
| 2 | SOLZ | Second-Order Laue Zone |
| ≥ 3 | HOLZ | Higher-Order Laue Zones |

Zones can be selected individually (`ZOLZ`, `FOLZ`, …), as a comma list
(`ZOLZ,FOLZ`), or as `ALL`.

### 2.2 Centering extinction rules

| Centering | Allowed condition | Typical lattice |
|---|---|---|
| P | all (h k l) | primitive |
| I | h + k + l even | BCC |
| F | h, k, l all same parity (all odd or all even) | FCC |
| C | h + k even | base-centered |
| R (hex setting) | −h + k + l ≡ 0 (mod 3) | rhombohedral, obverse setting |
| R (rhombohedral axes) | none (primitive basis) | trigonal, setting 2 |

### 2.3 Excitation error and thickness fringes

```
s ≈ g∥ + |g|²/(2k)            (Ewald sphere, small-angle form)
I(s, t) = sinc²(s·t) = [ sin(π·s·t) / (π·s·t) ]²      (NumPy sinc convention)
```

The thickness appears only through sinc²: the intensity envelope has its first zero at
`s = 1/t`, so the fringe period in s is Δs = 1/t. Example: t = 50 nm = 500 Å → 0.002 Å⁻¹.
Thin specimens (small t) give a broad envelope (more weakly excited spots survive);
thick specimens give a narrow one (cleaner, sparser patterns).

### 2.4 Intensity model and display

```
I_raw = sinc²(s·t) × ZONE_DAMPING
```

| Zone | Damping weight |
|---|---|
| ZOLZ | 1.00 |
| FOLZ | 0.30 |
| SOLZ | 0.15 |
| HOLZ | 0.05 |

Display pipeline: normalize by I_max → apply contrast power I^γ (γ default 0.60; γ < 1
visually lifts weak spots) → drop reflections below I_min (default 0.02) → map intensity
to marker alpha in the range [0.10, 1.00].

> **Scope note:** intensities in this version are **purely geometric** — atomic scattering
> factors and structure factors F(hkl) are **not** included (planned extension, §12).
> Spot *positions*, *indices*, *angles*, and *excitation errors* are exact within the
> stated model; only relative *brightness* is heuristic and should not be compared
> quantitatively with experiment.

### 2.5 Reference spots (indexing anchors)

Two reference reflections are selected from the **clean ZOLZ**, independent of the Ewald
correction and of intensities, so that indexing is reproducible between runs:

- ref2 ≠ ±ref1 (the Friedel pair is always avoided) and the two are non-collinear
  in the diffraction plane;
- (g1 × g2)·zone > 0 is enforced (right-handed indexing frame), by swapping only —
  never by blindly flipping a spot;
- after the display rotation, a ref2 lying above the +x axis is preferred;
- the search cap is hkl_cap = max(12, min(30, hkl_max)). If fewer than 3 clean ZOLZ
  reflections exist, the program raises an error — increase hkl_max.

The summary reports both references with their s values, the angle between them, and the
triple-product check.

---

## 3. Supported lattices — the 14 Bravais types

| # | Lattice | System | Centering |
|---|---|---|---|
| 1 | Cubic P | cubic | P |
| 2 | Cubic I | cubic | I (BCC) |
| 3 | Cubic F | cubic | F (FCC) |
| 4 | Tetragonal P | tetragonal | P |
| 5 | Tetragonal I | tetragonal | I |
| 6 | Orthorhombic P | orthorhombic | P |
| 7 | Orthorhombic C | orthorhombic | C |
| 8 | Orthorhombic I | orthorhombic | I |
| 9 | Orthorhombic F | orthorhombic | F |
| 10 | Hexagonal P | hexagonal | P |
| 11 | Trigonal R | trigonal | R (hex setting) or primitive rhombohedral |
| 12 | Monoclinic P | monoclinic | P |
| 13 | Monoclinic C | monoclinic | C |
| 14 | Triclinic P | triclinic | P |

- **Trigonal** asks for the setting: `1` = hexagonal axes (a, c; γ = 120°) or
  `2` = rhombohedral axes (a, α = β = γ).
- **Monoclinic** uses the unique-axis-β convention (α = γ = 90°, β free).
- All input cells are given as a, b, c in Å and angles in degrees.

---

## 4. Interactive inputs

| Prompt | Meaning | Default |
|---|---|---|
| Bravais choice (1–14) | lattice system + centering | — |
| a, b, c, angles | direct cell parameters | per system |
| TEM voltage (kV) | → λ, k | 200 |
| Zone axis [u v w] | beam direction | — |
| Laue zones | ZOLZ / FOLZ / SOLZ / HOLZ / list / ALL | ZOLZ |
| Nmax | max \|hu+kv+lw\| (auto from selection) | auto |
| Thickness t (nm) | sinc² thickness | 50 |
| I_min (0–1) | display cutoff on normalized intensity | 0.02 |
| Contrast γ | display shaping | 0.60 |
| hkl_max | search bound (±h, ±k, ±l) | 15 |
| Ewald correction | realistic s vs flat pattern | prompted |
| (h k l) labels | show indices on spots | yes |

---

## 5. Outputs

### 5.1 Book-like diffraction figure

- Black background ("textbook plate" style), square canvas
- Color-coded Laue zones: ZOLZ = white, FOLZ = lime, SOLZ = yellow, HOLZ = orange
- Reference spot 1 labeled in **cyan**, spot 2 in **magenta**
- Optional (h k l) labels on spots; central transmitted-beam marker
- Title: zone axis / centering / kV / thickness

<!-- ![Example SAED pattern](docs/example_fcc_011.png) -->

### 5.2 SAED SUMMARY (text report)

The console summary documents everything needed to reproduce the figure:

- lattice system, centering, full cell parameters
- zone axis, zones shown, Nmax, hkl_max
- voltage, λ, k, Ewald on/off
- thickness, intensity model, zone-damping weights, I_min, γ
- number of generated candidates vs plotted spots, I_max
- reference spots (h k l, s), the angle between them, (g1×g2)·zone check
- notes (e.g., a reference below the cutoff is still used for indexing)

---

## 6. Worked examples & validation benchmarks

Run these first — their geometry is textbook-known, so they double as sanity checks.
The cubic angle check is `cos∠(g1,g2) = (h1h2+k1k2+l1l2) / (|g1|·|g2|)`.

| Example | Cell | Zone | What you should see |
|---|---|---|---|
| FCC (austenite, a = 3.59 Å) | γ-Fe | [0 1 1] | 111-family, 200, 02̄2, 311…; ∠(200,111) = 54.74°, ∠(111,1̄11) = 70.53°, ∠(200,02̄2) = 90°; only all-odd / all-even hkl |
| BCC (ferrite, a = 2.866 Å) | α-Fe | [0 0 1] | square net; {110} family on the diagonals (45° from {200}); only h+k+l even |
| HCP (α-Ti, a = 2.951, c = 4.684 Å) | hexagonal | [0 0 1] (= [0001]) | hexagonal net of (100)-type nearest spots, 60°/120° angles |

If your simulated angles/geometry match these three, the geometric core is working.

---

## 7. Applications

1. **Teaching & courses** — generate any indexed SAED pattern on demand for lectures on
   crystallography, reciprocal space, and TEM diffraction (no photocopied atlases needed).
2. **Beam-time planning** — know in advance which reflections a planned zone axis shows
   and set tilt targets before touching the microscope.
3. **Indexing training & aid** — measure spot ratios and angles on an experimental pattern,
   reproduce the candidate geometry here, and match.
4. **Phase-identification practice** — spot-geometry fingerprints, e.g., FCC austenite vs
   BCC ferrite in dual-phase/TRIP steels.
5. **Thesis / paper / slide figures** — clean, consistent, indexed schematic patterns.
6. **Laue-zone self-study** — interactively see how the zone law separates ZOLZ from
   FOLZ/SOLZ/HOLZ, and how Ewald excitation selects spots.
7. **Manuscript sanity checks** — "which reflections appear in the ZOLZ of [011] FCC?"
   answered in seconds.
8. **Companion to deformation simulations** — for TEM-based TRIP/dual-phase work: predict
   the expected γ (FCC [011]) and α (BCC [001]) patterns before analyzing experimental SADs.

---

## 8. Practical tips

1. First session: ZOLZ + Ewald on + 200 kV + t = 50 nm + I_min = 0.02 + hkl_max = 15.
2. Ewald **off** gives the ideal "book" pattern (every ZOLZ spot at full excitation);
   Ewald **on** gives realistic excitation errors (weak spots fade, HOLZ rings appear).
3. Crowded pattern → raise I_min (0.05–0.10); too sparse → lower it.
4. γ < 1 lifts weak spots visually; γ > 1 emphasizes only the strongest spots.
5. Start from high-symmetry zone axes: cubic [001], [011], [111]; hexagonal [0001].
6. Compare **geometry and indices** with experiment — not absolute brightness (§2.4).
7. Pattern coordinates are in Å⁻¹ (g-space). To compare with film/camera distance:
   use R·d = λL with d = 2π/|g| from this tool.
8. If the program reports "Not enough ZOLZ reflections", increase hkl_max (e.g., 20).
9. Higher-order zones need larger hkl_max to be complete.
10. HOLZ rings are drawn on the flat tangent-plane projection; their curvature is
    approximate (§10).

---

## 9. Program structure

| Function | Purpose |
|---|---|
| `direct_basis_from_params` | (a,b,c,α,β,γ) → Cartesian basis vectors (all systems; rejects degenerate cells) |
| `reciprocal_basis` | a*, b*, c* (2π convention) |
| `g_vector_cart` | g(hkl) in Cartesian coordinates |
| `d_spacing` | d = 2π/\|g\| |
| `electron_wavelength_angstrom` | relativistic λ from kV |
| `zone_real_direction` | [u v w] → unit zone-axis vector |
| `make_plane_axes`, `g_perp`, `to_plane_xy`, `rotate_2d` | diffraction-plane geometry and display rotation |
| `allowed_by_centering` | P / I / F / C / R extinction rules |
| `laue_zone_label`, `parse_zone_selection`, `inferred_Nmax_from_selection` | Laue-zone bookkeeping |
| `generate_reflections_all` | candidate generation: centering + zone law + Ewald + intensity |
| `normalize_and_filter_by_intensity` | normalize, contrast γ, cutoff I_min |
| `choose_reference_spots_from_clean_zolz` | robust indexing reference pair |
| `angle_between_reflections` | true-metric angle (works for non-cubic cells) |
| `find_index_of_hkl` | locate a reflection in the kept list |
| `plot_saed_booklike` | book-style figure |
| `build_lattice_from_selection` | interactive 14-lattice builder |
| `BRAVAIS_14`, `ZONE_DAMPING`, style constants | configuration tables |

---

## 10. Limitations & simplifications

1. **Geometric intensity only** — no atomic form factors, no structure factor F(hkl),
   no chemistry dependence; relative brightness is heuristic (§2.4).
2. **Kinematic approximation** — no dynamical (multi-beam) scattering; real TEM
   intensities are dynamical.
3. **Single crystal, single zone axis** — no powder rings, no texture, no grain mixtures.
4. **No Kikuchi lines, no CBED discs, no beam precession.**
5. **Perfect, infinite crystal** — no defects, strain fields, or faulting; thickness
   enters only via the sinc² envelope.
6. **Flat-detector projection** — HOLZ ring curvature is approximate; the exact
   Ewald-sphere intersection is not rendered.
7. **Units** — the pattern is in Å⁻¹; no camera-length mm scale bar.
8. **Zone-damping weights** (1.0 / 0.3 / 0.15 / 0.05) are pedagogical, not
   physics-derived.

---

## 11. Items to verify before publication-grade use

1. Sign convention of the excitation error s against your reference text
   (e.g., Williams & Carter).
2. Consistency of the 2π convention (k = 2π/λ) across a*, k, and s.
3. Rhombohedral trigonal: the obverse hexagonal setting is implemented
   (−h+k+l ≡ 0 mod 3); the reverse setting differs.
4. Monoclinic: unique axis is β.
5. Thickness-fringe periodicity (Δs = 1/t) against a known thickness-fringe case.
6. Validate at least one pattern against an experimental SAED or a reference atlas.

---

## 12. Roadmap / possible extensions

1. **Structure factors** — atomic scattering factors (e.g., Doyle–Turner) plus basis
   atoms/Wyckoff positions → chemically meaningful intensities. *(top priority)*
2. Dynamical intensities (Bloch-wave or multislice) for selected zones.
3. Kikuchi-line overlay.
4. CBED / LACBED mode (discs with HOLZ lines).
5. Powder-ring / texture-arc mode for polycrystals.
6. Multi-phase side-by-side patterns (e.g., FCC γ + BCC α).
7. Batch/CLI mode with JSON/CSV export of the reflection table.
8. Camera-length calibration → mm scale bar.
9. Precession electron diffraction (PED) approximation.
10. Simple GUI (e.g., Streamlit) for classroom use.

---

## 13. Installation & usage

```bash
pip install numpy matplotlib
python saed_simulator.py
```

Then follow the interactive prompts (§4). A typical first session:
`Cubic F (FCC)` → a = 3.59 Å → 200 kV → zone `0 1 1` → ZOLZ → defaults → Ewald on.

---

## 14. References

1. D. B. Williams, C. B. Carter, *Transmission Electron Microscopy: A Textbook for
   Materials Science*, Springer.
2. B. Fultz, J. M. Howe, *Transmission Electron Microscopy and Diffractometry of
   Materials*, Springer.
3. C. Hammond, *The Basics of Crystallography and Diffraction*, IUCr Texts on
   Crystallography, Oxford University Press.
4. M. De Graef, M. E. McHenry, *Structure of Materials*, Cambridge University Press.
5. P. Hirsch et al., *Electron Microscopy of Thin Crystals*, Krieger.

---

## 15. License

MIT — see `LICENSE`.
