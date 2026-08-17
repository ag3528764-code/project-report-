# Groundwater Level Prediction System
### Comprehensive Technical Architecture & Physical Modeling Report

---

## SECTION 1: Hydro-Geological Subsurface Physics & Data Mechanics

### 1.1 Subsurface Layer Architecture & Hydro-Spatial Dynamics
Groundwater exists within three distinct subsurface hydro-geological zones, each imposing unique temporal boundaries and dynamic constraints on predictive machine learning models:

<img width="601" height="340" alt="Picture1" src="https://github.com/user-attachments/assets/1f713266-041a-4e49-a85e-bdff9634025a" />




* **Unconfined Aquifer (Phreatic / Water Table Layer):**
  * **Physical Behavior:** Directly exposed to atmospheric condition changes via unsaturated soil zone infiltration ($f_i$). Lacks an upper impermeable boundary.
  * **ML Architecture Impact:** High temporal sensitivity. Models must account for short lag times ($1 \le \tau \le 15\text{ days}$) relative to surface precipitation events ($P$). Exhibits high dynamic variance.
* **Confined Aquifer (Artesian Reservoir):**
  * **Physical Behavior:** Trapped between impermeable aquitard boundaries under hydrostatic pressure. Primary recharge originates from distant lateral catchment basins rather than direct vertical infiltration.
  * **ML Architecture Impact:** Long temporal memory. Models require long lag windows ($30 \le \tau \le 180\text{ days}$) via Recurrent/Transformer attention mechanisms. Movement is dictated by hydraulic head gradients ($\Delta h$) rather than local volumetric infiltration.
* **Aquitard Boundary (Confining Bed):**
  * **Physical Behavior:** Composed of dense clays or solid igneous rock matrices with extremely low hydraulic conductivity ($K \approx 10^{-8}\text{ m/s}$). Acts as a semi-impermeable flow barrier.
  * **ML Architecture Impact:** Functions as a boundary condition constraint limiting vertical flux between unconfined and confined model states.

---

### 1.4 Spatial-Temporal Data Mechanics ($T \times \text{Lat} \times \text{Lon}$ NetCDF4 Data Cubes)

![Picture 2: NetCDF4 Spatio-Temporal Data Cube Tensor Layout](<img width="601" height="416" alt="Picture2" src="https://github.com/user-attachments/assets/34670a55-0615-4dc8-8e52-78ed19009b2e" />
)

#### Tabular Feature Schema (Model Input Tensor Schema)
Below is the processed tabular feature schema extracted from spatial NetCDF4 cubes and ground truth sensors:

![Picture 3: Tabular Feature Schema derived from Spatial Cubes](<img width="602" height="221" alt="Picture3" src="https://github.com/user-attachments/assets/df26b0c9-36eb-44cc-8dd3-e4274cff93e9" />
)

![Picture 4: Synthetic Dataset Sample - First 5 Rows](<img width="602" height="143" alt="Picture4" src="https://github.com/user-attachments/assets/fc459a3b-38a2-4c76-9d6a-1e40648aa32c" />
)

---

## SECTION 2: Multi-Modal Satellite Data & Ingestion Pipeline

### 2.1 Data Ingestion Architecture
![Picture 5: Multi-Modal Satellite Data Ingestion Pipeline Architecture](<img width="525" height="418" alt="Picture5" src="https://github.com/user-attachments/assets/ac437172-9081-4cc6-994b-8693a7430f49" />
)

---

### 2.2 Deep-Dive on Remote Sensing Modalities

#### A. NASA GRACE & GRACE-FO (Gravity Recovery Satellites)
![Picture 6: NASA GRACE Inter-Satellite Laser Ranging Mechanism](<img width="1378" height="576" alt="Picture6" src="https://github.com/user-attachments/assets/9114e8d5-635c-4118-9d55-ef5323cabf29" />
)

#### B. ISRO Bhuvan & MOSDAC (Land Use / Land Cover - LULC)
![Picture 7: ISRO Bhuvan / MOSDAC Land Use Land Cover (LULC) Map](<img width="1378" height="152" alt="Picture7" src="https://github.com/user-attachments/assets/186d7e0b-364f-441f-a7a6-12a3e8eac9b0" />
)

#### C. Sentinel-1 Synthetic Aperture Radar (SAR)
![Picture 8: Sentinel-1 SAR C-Band Dielectric Backscatter Response](<img width="1378" height="448" alt="Picture8" src="https://github.com/user-attachments/assets/af989ef7-78a6-4ada-a912-7b97ab2f23d5" />
)

#### D. CGWB Ground Truth Ingestion
![Picture 9: Depth to Absolute Hydraulic Head Elevation](<img width="1378" height="752" alt="Picture9" src="https://github.com/user-attachments/assets/9d0ef269-79f2-453a-bc2e-e5878a3fcea5" />
)

---

### 2.3 Spatial Harmonization Engine
![Picture 10: Spatial Harmonization & Mesh Resampling Flow](<img width="1378" height="367" alt="Picture10 1" src="https://github.com/user-attachments/assets/3fbd94f6-935a-4ed4-addd-4254d348596a" />
)
<img width="1378" height="359" alt="Picture10" src="https://github.com/user-attachments/assets/0e9961cc-a8fb-4d11-aad6-214f7b2dc954" />


![Picture 11: Harmonized Feature Matrix Schema](<img width="602" height="225" alt="Picture11" src="https://github.com/user-attachments/assets/8058ba9c-2967-40bb-8728-d3efd5c36d6e" />
)

![Picture 12: Synthetic Ingestion Data Representation](<img width="602" height="101" alt="Picture12" src="https://github.com/user-attachments/assets/c8759f93-9a3e-4fb5-890c-0c7851957001" />
)

---

## SECTION 3: Feature Engineering & Temporal Lag Pipeline

![Picture 13: DEM-Derived Topographic & Drainage Features](<img width="1375" height="682" alt="Picture13" src="https://github.com/user-attachments/assets/9a4d8c66-ed13-4689-ac0b-a2ab2b9bb4ae" />
)

![Picture 14: Percolation Delay & Temporal Lag Tensor Formation](<img width="602" height="164" alt="Picture14" src="https://github.com/user-attachments/assets/2a1b1fcf-89e7-47a7-a2d8-33607cb2c0c2" />
)

![Picture 15: Feature Engineering Output Schema](./images/Picture15.png)

![Picture 16: Synthetic Dataset Sample Window](./images/Picture16.png)

---

## SECTION 4: AI/ML Architecture Strategy

![Picture 17: Multi-Tier AI/ML Architecture Framework](./images/Picture17.png)

![Picture 18: ConvLSTM Architecture and Spatial Convolutional Gates](./images/Picture18.png)

![Picture 19: Spatio-Temporal Graph Neural Network Topology](./images/Picture19.png)

![Picture 20: Comprehensive Architecture Comparison Table](./images/Picture20.png)

---

## SECTION 5: Physics-Informed Neural Networks (PINNs)

![Picture 21: Standard Data-Driven vs PINN Predictions](./images/Picture21.png)

![Picture 22: Model Evaluation Metrics & PINN Optimization Comparison](./images/Picture22.png)

---

## SECTION 6: Geomorphology-Driven Hydrogeology & System Architecture Blueprint

![Picture 23: Geomorphic Classification Map & Cross-Sections](./images/Picture23.png)

![Picture 24: Upgraded System Architecture Blueprint](./images/Picture24.png)

![Picture 25: Complete End-to-End System Integration Schema](./images/Picture25.png)

---

## SECTION 7: Geospatial Dashboard Architecture & Design Specification

![Picture 26: Module Scope and Phase Matrix Table](./images/Picture26.png)

![Picture 27: Geospatial Web-GIS UI/UX Wireframe Layout](./images/Picture27.png)

![Picture 28: Full-Stack Web-GIS Tech Stack Integration Schema](./images/Picture28.png)

![Picture 29: End-to-End User Operational Workflow Diagram](./images/Picture29.png)
