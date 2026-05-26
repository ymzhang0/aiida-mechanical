# Welcome to `aiida-mechanical`

`aiida-mechanical` is an AiiDA plugin for managing elastic calculations using thermo_pw and stacking fault energy calculations using Quantum ESPRESSO.

---

## 🚀 Key Features

### 1. Elastic Constants (`thermo_pw` module)
This module automates calculations of:
* Elastic constants and shear/bulk moduli.


### 2. Dislocation Workflows (`dislocation` module)
Integrated to evaluate planar defects and generalized stacking fault energy (GSFE) surfaces:
* **GSFE**: Computes generalized stacking fault energy curves along customizable gliding planes.
* **Surface Energy**: Computes surface energy.


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
* [Quick Start Guide](quickstart.md) — Get up and running with example scripts for `thermo_pw` and `gsfe` workflows.
* [Developer Guide](developer.md) — Setup local editable installations, pre-commit styling, and testing.
