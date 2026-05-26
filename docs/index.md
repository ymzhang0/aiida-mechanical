# Welcome to `aiida-mechanical`

`aiida-mechanical` is a comprehensive AiiDA plugin designed for managing, running, and analyzing mechanical and thermodynamic properties of materials. It integrates high-throughput workflow engines for crystal elasticity, structural shear, and defect modeling.

---

## 🚀 Key Features

### 1. Thermodynamic & Elasticity Properties (`thermo_pw` module)
Derived and extended from Quantum ESPRESSO's `thermo_pw` project, this module automates high-throughput calculations of:
* Elastic constants and shear/bulk moduli.
* Phonon-driven thermodynamic properties.
* Automatic spacegroup analysis and Bravais lattice generation.

### 2. Defect & Dislocation Workflows (`dislocation` module)
Integrated to evaluate planar defects and generalized stacking fault energy (GSFE) surfaces:
* **GSFE & USFE WorkChains**: Computes stable and unstable stacking fault energy curves along customizable gliding planes.
* **Relaxation & Rigid Layer Shear**: Integrates rigid-layer shear sliding and full k-point adaptive relaxation.
* **Twinning & Surface Energy**: Automation of crystal twinning energies and surface cleavage energies.

---

## 🏗️ Registered Entry Points

Here are the primary entry points registered under the `mechanical` namespace:

### Workflows
* `mechanical.thermo_pw.base` — Base workflows for elastic/phonon engines.
* `mechanical.dislocation.gsfe` — Generalized Stacking Fault Energy WorkChain.
* `mechanical.dislocation.gsfe_relax` — Relaxed Generalized Stacking Fault Energy WorkChain.
* `mechanical.dislocation.isfe` — Inherent Stacking Fault Energy WorkChain.
* `mechanical.dislocation.esfe` — Empirical Stacking Fault Energy WorkChain.
* `mechanical.dislocation.usfe` — Unstable Stacking Fault Energy WorkChain.
* `mechanical.dislocation.twinning` — Twinning energy calculation WorkChain.

### Custom Stacking Fault Data Models
* `mechanical.dislocation.cleavaged_structure` — Managed slab structure with customized vacuum padding.
* `mechanical.dislocation.faulted_structure` — Shear-faulted crystal system containing glide-plane definitions.

---

## 📖 Quick Links
* [Developer Guide](developer.md) — Setup local editable installations, pre-commit styling, and testing.
