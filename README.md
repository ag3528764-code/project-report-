# Groundwater Level Prediction System
### Comprehensive Technical Architecture & Physical Modeling Report

---

## Table of Contents
1. [Hydro-Geological Subsurface Physics & Data Mechanics](#section-1-hydro-geological-subsurface-physics--data-mechanics)
2. [Multi-Modal Satellite Data & Ingestion Pipeline](#section-2-multi-modal-satellite-data--ingestion-pipeline)
3. [Feature Engineering & Temporal Lag Pipeline](#section-3-feature-engineering--temporal-lag-pipeline)
4. [AI/ML Architecture Strategy](#section-4-aiml-architecture-strategy)
5. [Physics-Informed Neural Networks (PINNs)](#section-5-physics-informed-neural-networks-pinns)
6. [Geomorphology-Driven Hydrogeology & System Architecture Blueprint](#section-6-geomorphology-driven-hydrogeology--system-architecture-blueprint)
7. [Geospatial Dashboard Architecture & Design Specification](#section-7-geospatial-dashboard-architecture--design-specification)

---

## SECTION 1: Hydro-Geological Subsurface Physics & Data Mechanics

### 1.1 Subsurface Layer Architecture & Hydro-Spatial Dynamics
Groundwater exists within three distinct subsurface hydro-geological zones, each imposing unique temporal boundaries and dynamic constraints on predictive machine learning models:

![Image 1: Subsurface Aquifer Layering and Hydrological Dynamics](./images/image1.png)

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

### 1.2 Soil Physical Properties & Static Feature Engineering
The static soil matrix dictates the rate and capacity of groundwater movement, serving as critical auxiliary features for the model:

* **Porosity ($\phi$):** The ratio of void space volume ($V_{\text{voids}}$) to total volume ($V_{\text{total}}$), setting the maximum volumetric storage capacity:
  $$\phi = \frac{V_{\text{voids}}}{V_{\text{total}}}$$

* **Hydraulic Conductivity ($K$):** Measures fluid transmissivity through porous media ($\text{m/day}$) according to Darcy's Law ($Q = K \cdot A \cdot \frac{\Delta h}{L}$):
  $$K = \frac{Q \cdot L}{A \cdot \Delta h}$$
  * **High $K$ (Gravel/Sand):** Fast infiltration; shorter lookback windows required.
  * **Low $K$ (Dense Clay):** Slow infiltration; extended temporal sequence length required.

* **Specific Yield ($S_y$):** The ratio of water volume drained by gravity to total volume ($0.05 \le S_y \le 0.30$). Acts as a scalar constant converting volumetric storage changes ($\Delta S$) to hydraulic head changes ($\Delta h$).

---

### 1.3 Mass Conservation & Physical Constraints
To prevent non-physical predictions, model formulations respect the hydrological mass balance equation:

$$\Delta S = (P + R_{\text{canal}} + Q_{\text{in}}) - (ET + Q_{\text{runoff}} + W_{\text{extraction}} + Q_{\text{out}})$$

**Where:**
* **Inflow (+):** Precipitation ($P$), Canal Infiltration ($R_{\text{canal}}$), Lateral Inflow ($Q_{\text{in}}$)
* **Outflow (-):** Evapotranspiration ($ET$), Surface Runoff ($Q_{\text{runoff}}$), Borewell Pumping ($W_{\text{extraction}}$), Lateral Outflow ($Q_{\text{out}}$)

The resulting change in water table head ($\Delta h$) over time step $\Delta t$ is strictly constrained by:
$$\Delta h = \frac{\Delta S}{S_y}$$

---

### 1.4 Spatial-Temporal Data Mechanics ($T \times \text{Lat} \times \text{Lon}$ NetCDF4 Data Cubes)

![Image 2: NetCDF4 Spatio-Temporal Data Cube Tensor Layout](./images/image2.png)

The tensor layout $[T \times \text{Lat} \times \text{Lon}]$ structures spatial-temporal information as follows:
* **$T$ (Temporal Axis / Sequence Length):** Represents time steps (e.g., daily or monthly observation snapshots). Sliding windows along $T$ construct input sequences for time-series forecasting.
* **$\text{Lat} \times \text{Lon}$ (Spatial Grid / Spatial Dimensions):** A 2D grid where each cell $(i,j)$ represents a spatial coordinate index corresponding to physical ground coverage.

#### Tabular Feature Schema (Model Input Tensor Schema)
Below is the processed tabular feature schema extracted from spatial NetCDF4 cubes and ground truth sensors:

![Image 3: Tabular Feature Schema derived from Spatial Cubes](./images/image3.png)

![Image 4: Synthetic Dataset Sample - First 5 Rows](./images/image4.png)

---

## SECTION 2: Multi-Modal Satellite Data & Ingestion Pipeline

### 2.1 Data Ingestion Architecture
The ingestion engine processes diverse satellite data formats, harmonizing spatial resolutions and projection systems before feeding target tensors into spatial-temporal neural network backbones:

![Image 5: Multi-Modal Satellite Data Ingestion Pipeline Architecture](./images/image5.png)

---

### 2.2 Deep-Dive on Remote Sensing Modalities

#### A. NASA GRACE & GRACE-FO (Gravity Recovery Satellites)
**Physical Mechanism:** Operates under Newton's Law of Universal Gravitation ($F = G \frac{m_1 m_2}{r^2}$). Dual co-orbiting satellites ("Tom" and "Jerry") maintain continuous K-Band / Laser Ranging inter-satellite distance measurements ($\Delta d$). Spatial variations in underground mass ($\Delta m$) alter local gravitational acceleration, accelerating the lead satellite and creating sub-micron distance variations.

![Image 6: NASA GRACE Inter-Satellite Laser Ranging Mechanism](./images/image6.png)

* **Data Format:** NetCDF4 (`.nc`) Data Cubes ($T \times \text{Lat} \times \text{Lon}$).
* **Output Variable:** Terrestrial Water Storage Anomaly ($\Delta TWS$) measured in Liquid Water Equivalent thickness ($\text{cm}$).

#### B. ISRO Bhuvan & MOSDAC (Land Use / Land Cover - LULC)
**Physical Mechanism:** Categorizes surface cover types to infer hydraulic run-off coefficients ($C_{\text{runoff}}$) and recharge potential:
* **Built-Up / Urban (Concrete):** High runoff coefficient ($C_{\text{runoff}} \approx 0.85$), near-zero vertical infiltration.
* **Agricultural Land:** High extraction via tubewells coupled with irrigation return flow seepage.
* **Forest Canopy:** High canopy interception and soil infiltration, low surface runoff.

![Image 7: ISRO Bhuvan / MOSDAC Land Use Land Cover (LULC) Map](./images/image7.png)

#### C. Sentinel-1 Synthetic Aperture Radar (SAR)
**Physical Mechanism:** Active microwave sensing using C-Band frequencies ($\lambda = 5.54\text{ cm}$) capable of penetrating dense cloud cover during monsoon periods.

**Dielectric Constant Physics:** Dry soil exhibits a low dielectric constant ($\epsilon_r \approx 3\text{--}5$), whereas liquid water exhibits a high dielectric constant ($\epsilon_r \approx 80$). Backscatter intensity ($\sigma^0$) directly correlates with volumetric soil moisture within the top surface layer ($0\text{--}5\text{ cm}$).

![Image 8: Sentinel-1 SAR C-Band Dielectric Backscatter Response](./images/image8.png)

#### D. CGWB Ground Truth Ingestion (Target Variable Calculation)
Groundwater observation wells feature piezoelectric pressure sensors recording water table depth below ground level ($\text{Depth}_{\text{measured}}$ in $\text{m}_{\text{bgl}}$).

To model hydraulic pressure gradients across variable topographies (e.g., elevated terrain vs. low-lying valleys), raw depth values are converted to absolute Hydraulic Head Elevations ($H_{\text{water}}$) above Mean Sea Level (AMSL):

$$H_{\text{water}} = Z_{\text{surface\_DEM}} - \text{Depth}_{\text{measured}}$$

![Image 9: Transformation of Depth Below Ground Level to Absolute Hydraulic Head Elevation](./images/image9.png)

Without converting to absolute Hydraulic Head ($H_{\text{water}}$), deep learning architectures and spatial graph networks cannot calculate correct hydraulic gradients ($\Delta H / L$) governing subsurface flow direction under Darcy's Law.

---

### 2.3 Spatial Harmonization Engine
Inverting spatial datasets of varying spatial resolutions and projection formats into a structured neural tensor requires two primary spatial harmonization stages:

![Image 10: Spatial Harmonization & Mesh Resampling Flow](./images/image10.png)

* **Step 1: Coordinate Reference System (CRS) Reprojection:**  
  Raw satellite data is distributed in Geographic Coordinate Systems (EPSG:4326 - WGS84) where spatial units are expressed in angular degrees. Because distance per degree varies across latitudes, data is reprojected into a metric Projected Coordinate System (UTM Zone 43N / EPSG:32643 for Central India), establishing grid units strictly in meters.
* **Step 2: Multi-Scale Spatial Resampling (Mesh Gridding):**  
  The region of interest is discretized into a uniform $1\text{ km} \times 1\text{ km}$ master mesh:
  * **Coarse Datasets (GRACE $110\text{ km} \rightarrow 1\text{ km}$):** Interpolated via Ordinary Kriging to construct spatially continuous fields across $1\text{ km}$ grid cells.
  * **Fine Datasets (Sentinel $10\text{ m} \rightarrow 1\text{ km}$):** Aggregated via Zonal Mean reduction across the 10,000 constituent $10\text{ m}$ sub-pixels falling within each target $1\text{ km}$ grid cell.

#### Data Schema & Feature Pipeline Table
![Image 11: Harmonized Feature Matrix Schema (1 km x 1 km Grid Cell Vector)](./images/image11.png)

![Image 12: Synthetic Ingestion Data Representation (Sample Batch Output)](./images/image12.png)

---

## SECTION 3: Feature Engineering & Temporal Lag Pipeline

### 3.1 Topographic Geometry & Spatial Drainage Engineering
Atmospheric precipitation either runs off surface terrain or infiltrates into the subsurface based on localized terrain geometry. These static features are derived from NASA SRTM Digital Elevation Models (DEM):

![Image 13: DEM-Derived Topographic & Drainage Features](./images/image13.png)

* **Digital Elevation ($Z$):** Surface height above Mean Sea Level (AMSL) in meters, providing baseline vertical spatial indexing.
* **Slope ($\beta$) & Aspect:**
  * **Slope ($\beta$):** Local terrain gradient measured in degrees or radians:
    $$\beta = \arctan \sqrt{\left(\frac{\partial Z}{\partial x}\right)^2 + \left(\frac{\partial Z}{\partial y}\right)^2}$$
  * **Aspect:** Directional orientation of the terrain slope ($0^\circ$ to $360^\circ$), influencing evapotranspiration solar radiation exposure.
* **Topographic Wetness Index ($\text{TWI}$):** A physics-based metric defining spatial water accumulation potential:
  $$\text{TWI} = \ln\left(\frac{a}{\tan \beta}\right)$$
  * **Where:**
    * $a$: Specific catchment area ($\text{m}^2/\text{m}$), representing upslope contributing area per unit contour length.
    * $\beta$: Local slope angle in radians.
    * **Physical Interpretation:** High $\text{TWI}$ values signify low-slope catchment depressions with high infiltration potential; low $\text{TWI}$ values indicate steep ridges where runoff dominates over recharge.

---

### 3.2 Dynamic Hydrological & Extraction Proxies
Surface vegetation and water body fluctuations serve as indirect proxies for anthropogenic extraction and surface-subsurface hydrologic exchanges:

* **Normalized Difference Vegetation Index ($\text{NDVI}$):**
  $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
  Measures photosynthetic canopy vigor. In agricultural regions, high peak $\text{NDVI}$ correlates directly with intensive crop cultivation, acting as a dynamic proxy for groundwater pumping and borewell extraction.

* **Normalized Difference Water Index ($\text{NDWI}$):**
  $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
  Identifies surface water bodies (lakes, reservoirs, rivers) that serve as localized recharge boundaries.

* **Effective Dynamic Recharge Index ($R_i(t)$):**
  Calculates actual effective vertical water entry into the topsoil matrix by coupling precipitation ($P(t)$) with land cover runoff coefficients ($C_{\text{runoff}}$) and saturated soil hydraulic conductivity ($K_{\text{soil}}$):
  $$R_i(t) = P(t) \cdot (1 - C_{\text{runoff}}) \cdot K_{\text{soil}}$$

---

### 3.3 Temporal Memory & Multi-Scale Lag Tensors
Precipitation events do not instantaneously replenish deep aquifers. Water must percolate through heterogeneous unsaturated soil layers (vadose zone), creating variable time delays ($\tau$) ranging from days to several months depending on soil depth and hydraulic conductivity.

To capture these percolation delays, input feature vectors incorporate multi-scale temporal lag matrices:

$$\mathbf{X}_{\text{Dynamic}}(t) = \begin{bmatrix} P(t) & P(t-7) & P(t-15) & P(t-30) & P(t-60) & P(t-90) & P(t-180) \end{bmatrix}^T$$

![Image 14: Percolation Delay & Multi-Scale Temporal Lag Tensor Formation](./images/image14.png)

#### Data Schema & Feature Transformation Table
![Image 15: Feature Engineering Output Schema (Input Matrix X)](./images/image15.png)

![Image 16: Synthetic Dataset Sample (Feature Matrix Window)](./images/image16.png)

---

## SECTION 4: AI/ML Architecture Strategy

### 4.1 Model Architectural Paradigm
![Image 17: Multi-Tier AI/ML Architecture Framework](./images/image17.png)

---

### 4.2 Tier 1: Baseline Benchmark via Geospatial Gradient Boosting
Before deploying deep neural networks, a Geospatial XGBoost / LightGBM model establishes baseline error metrics ($\text{RMSE}, R^2$).

* **Data Structure:** Tabular schema indexed by $(\text{Well\_ID}, \text{Timestamp})$.
* **Input Vector:**
  $$\mathbf{x}_i = \left[ Z, \beta, \text{TWI}, K_{\text{soil}}, \text{NDVI}(t), \text{NDWI}(t), P(t), P(t-7), \dots, P(t-180), \text{GRACE}_{\text{anomaly}} \right]$$
* **Model Interpretability (SHAP Integration):**
  SHAP values quantify feature contributions to predictions using game theory principles:
  $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$
  Where $F$ represents the total feature set, and $S$ is a feature subset excluding index $i$.

---

### 4.3 Tier 2: ConvLSTM (Convolutional Long Short-Term Memory)
Regular LSTMs handle sequential data using dense vector multiplications, ignoring 2D grid structures. ConvLSTM addresses this by using 2D spatial convolutions ($*$) inside its gating mechanisms.

![Image 18: ConvLSTM Architecture and Spatial Convolutional Gates](./images/image18.png)

#### ConvLSTM Mathematical Formulation
$$i_t = \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + W_{ci} \circ C_{t-1} + b_i)$$
$$f_t = \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + W_{cf} \circ C_{t-1} + b_f)$$
$$C_t = f_t \circ C_{t-1} + i_t \circ \tanh(W_{xc} * X_t + W_{hc} * H_{t-1} + b_c)$$
$$o_t = \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + W_{co} \circ C_t + b_o)$$
$$H_t = o_t \circ \tanh(C_t)$$

**Operator Definitions:**
* $*$: 2D Spatial Convolution operator (extracts regional neighbor dynamics).
* $\circ$: Hadamard (element-wise) product operator.
* $X_t, H_t, C_t$: Input tensor, Hidden state tensor, and Cell state tensor in 3D grid shape ($C \times H \times W$).

---

### 4.4 Tier 3: Spatio-Temporal Graph Neural Networks (ST-GNN)
Real-world aquifers do not conform to uniform spatial grids. Geological formations function as irregular networks where flow depends on subterranean permeability and topography.

![Image 19: Spatio-Temporal Graph Neural Network Topology](./images/image19.png)

#### Graph Formulation
$$\mathcal{G} = (V, E, A)$$

* **Nodes ($V$):** $N$ discrete observation wells or sensor nodes.
* **Edges ($E$):** Underground hydraulic connectivity paths between wells.
* **Hydro-Geological Weighted Adjacency Matrix ($A$):**
  $$A_{ij} = \exp\left( -\frac{d_{ij}^2}{\sigma^2} \right) \times \frac{K_{ij}}{\max(K)}$$
  **Where:**
  * $d_{ij}$: Euclidean distance between well $i$ and well $j$.
  * $K_{ij}$: Mean subterranean soil permeability factor between nodes $i$ and $j$.
  * $\sigma^2$: Distance threshold variance scaling parameter.

#### Graph Convolutional Message Passing Layer
$$H^{(l+1)} = \sigma\left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

Where $\tilde{A} = A + I_N$ is the adjacency matrix with self-loops, $\tilde{D}$ is its diagonal degree matrix, and $W^{(l)}$ represents trainable weight parameters.

---

### 4.5 Comprehensive Model Architecture Comparison
![Image 20: Comprehensive Architecture Comparison Table](./images/image20.png)

---

## SECTION 5: Physics-Informed Neural Networks (PINNs)

### 5.1 Limitations of Purely Data-Driven Architectures
Standard Neural Networks optimized exclusively on Mean Squared Error ($\mathcal{L}_{\text{Data}}$) lack intrinsic knowledge of mass conservation. Under sparse sensor distributions or extreme climatic shifts, pure statistical models can predict non-physical outcomes—such as increasing head elevation during zero precipitation coupled with heavy extraction.

![Image 21: Comparison of Standard Data-Driven vs PINN Predictions](./images/image21.png)

---

### 5.2 Governing Physics: 2D Unconfined Boussinesq Equation
Transient subterranean fluid flow in a horizontal, unconfined homogeneous aquifer is governed by the non-linear 2D Boussinesq Partial Differential Equation:

$$S_y \frac{\partial h}{\partial t} = \frac{\partial}{\partial x} \left( K_x h \frac{\partial h}{\partial x} \right) + \frac{\partial}{\partial y} \left( K_y h \frac{\partial h}{\partial y} \right) + R(t) - W(t)$$

**Variable Definitions:**
* $h(x,y,t)$: Predicted piezometric hydraulic head level ($\text{m}$).
* $S_y$: Specific yield coefficient of the porous geological medium (dimensionless).
* $K_x, K_y$: Saturated hydraulic conductivity along directional axes ($\text{m/day}$).
* $R(t)$: Net effective dynamic recharge flux ($\text{m/day}$).
* $W(t)$: Volumetric groundwater extraction rate ($\text{m/day}$).

Under isotropic conductivity assumptions ($K_x = K_y = K$) and linearized local flow assumptions, the PDE Residual function $f(x,y,t)$ is defined as:

$$f := S_y \frac{\partial \hat{h}}{\partial t} - K \left[ \left(\frac{\partial \hat{h}}{\partial x}\right)^2 + \hat{h} \frac{\partial^2 \hat{h}}{\partial x^2} + \left(\frac{\partial \hat{h}}{\partial y}\right)^2 + \hat{h} \frac{\partial^2 \hat{h}}{\partial y^2} \right] - R(t) + W(t) = 0$$

**Physical Validity:**
* If $f(x,y,t) = 0$, the predicted head $\hat{h}$ strictly respects physical mass balance.
* If $f(x,y,t) \neq 0$, the network penalizes the deviation via $\mathcal{L}_{\text{Physics}} = \frac{1}{N_{\text{colloc}}} \sum |f(x,y,t)|^2$.

---

### 5.3 Loss Function Optimization Mechanics
The network is trained end-to-end using a composite multi-objective regularization framework:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Data}} + \lambda \cdot \mathcal{L}_{\text{Physics}}$$

$$\mathcal{L}_{\text{Data}} = \frac{1}{N_{\text{obs}}} \sum_{i=1}^{N_{\text{obs}}} \left( h(x_i, y_i, t_i) - \hat{h}(x_i, y_i, t_i) \right)^2$$

$$\mathcal{L}_{\text{Physics}} = \frac{1}{N_{\text{pde}}} \sum_{j=1}^{N_{\text{pde}}} \left| f(x_j, y_j, t_j) \right|^2$$

Where $\lambda \in [0.1, 10.0]$ is an adaptive regularization hyperparameter balancing empirical loss against PDE physical compliance.

---

### 5.4 Model Evaluation Metrics & PINN Paradigm Comparison
![Image 22: Model Evaluation Metrics & PINN Optimization Comparison](./images/image22.png)

---

## SECTION 6: Geomorphology-Driven Hydrogeology & System Architecture Blueprint

### 6.1 Geomorphic Unit Classification & Hydrogeological Potential
Landforms digitized via satellite remote sensing (e.g., ISRO Bhuvan, USGS Landsat) are classified into distinct hydro-geomorphic units based on permeable storage capacity and infiltration dynamics:

![Image 23: Geomorphic Classification Map & Cross-Sections](./images/image23.png)

* **Alluvial Plains:** Smooth, river-deposited active fluvial zones composed of sand, gravel, and silt.
  * *Hydrological Impact:* Very High Permeability; rapid vertical percolation into unconfined aquifers.
  * *Feature Weighting:* $+0.9$ (Positive Recharge Bias).
* **Valley Fills:** Low-lying structural depressions surrounded by elevated terrain, accumulating thick weathered debris.
  * *Hydrological Impact:* Natural catchment traps; sustained regional groundwater recharge.
  * *Feature Weighting:* $+0.8$ (High Accumulation Potential).
* **Pediments / Pediplains:** Gentle, sloping rock cut surfaces at the base of receding hills.
  * *Hydrological Impact:* Variable; depends heavily on local overburden weathering depth ($10\text{ m} \text{--} 30\text{ m}$).
  * *Feature Weighting:* $+0.4$ to $+0.6$ (Conditional Potential).
* **Denudational / Structural Hills:** Steep, un-weathered crystalline rock mass elevations.
  * *Hydrological Impact:* Very Low Infiltration; dominated by rapid surface runoff ($>95\%$ loss).
  * *Feature Weighting:* Negative Recharge Weight (Runoff Collector).

---

### 6.2 Quantitative Spatial Geomorphology Inputs
To feed machine learning models (XGBoost, ST-GNN, PINN), spatial geomorphology is parameterized into continuous quantitative raster matrices:

* **Lineament Density Index ($\text{LDI}$):**
  Quantifies secondary fracture networks and fault lines created by tectonic activity in hard rock terrains:
  $$\text{LDI} = \frac{\sum_{k=1}^M L_k}{A_{\text{cell}}}$$
  Where $L_k$ is the physical length of lineament segment $k$ within grid cell area $A_{\text{cell}}$ ($\text{km/km}^2$). Higher density correlates directly with secondary bedrock storage capacity.

* **Drainage Density ($\text{DD}$):**
  Measures the total stream length per unit catchment area:
  $$\text{DD} = \frac{\sum_{m=1}^N S_m}{A_{\text{cell}}}$$
  High drainage density indicates impermeable surface soil with high runoff rates, whereas low drainage density points to permeable surface materials promoting infiltration.

* **Topographic Wetness Index ($\text{TWI}$):**
  Combines upslope contributing catchment area ($a$) with local slope gradient ($\beta$):
  $$\text{TWI} = \ln\left(\frac{a}{\tan \beta}\right)$$
  High $\text{TWI}$ values correspond to flat, low-lying alluvial depressions prone to water accumulation; low $\text{TWI}$ values denote steep, well-drained ridges.

---

### 6.3 Upgraded System Architecture Blueprint
![Image 24: Upgraded System Architecture Blueprint](./images/image24.png)

![Image 25: Complete End-to-End System Integration Schema](./images/image25.png)

---

## SECTION 7: Geospatial Dashboard Architecture & Design Specification

The Web-GIS Geospatial Dashboard integrates subsurface physics, machine learning inference engines (XGBoost, Spatio-Temporal GNN, PINN), and multi-modal spatial vectors into a unified operational command center. It translates complex hydrogeological models into actionable intelligence for municipal authorities, hydrogeologists, and local planners.

### 7.1 Scope & Module Classification Matrix
The platform implementation follows a two-stage deployment model. Core predictive intelligence modules are active in the Pilot Phase (focusing on the Berasia block/ward administrative boundaries), while complex simulation modules are provisioned as Locked/Coming Soon for the Full-Scale Phase.

![Image 26: Module Scope and Phase Matrix Table](./images/image26.png)

---

### 7.2 Interactive Layout & UI/UX Wireframe Structure
The dashboard interface uses a multi-pane responsive layout designed for high spatial awareness and minimal cognitive overhead during spatial decision-making:

![Image 27: Geospatial Web-GIS UI/UX Wireframe Layout](./images/image27.png)

---

### 7.3 Deep Breakdown of Deliverables & Functional Modules

#### A. Groundwater Prospect Assessment Panel
* **Objective:** Identifies optimal zones for new groundwater abstraction while minimizing dry-drilling failure risks.
* **Layer Rendering:** Multi-resolution Vector Polygon Choropleth / Raster Heatmap.
* **Prospect Classification Scale:**
  * **Poor (Red):** Success Score $<30\%$ | Crystalline bedrock, low fracturing, low TWI.
  * **Moderate (Orange):** Success Score $30\%\text{--}60\%$ | Weathered pediment zones, moderate overburden.
  * **Good (Light Green):** Success Score $60\%\text{--}85\%$ | Moderate alluvial coverage, nearby lineaments.
  * **Excellent (Dark Green):** Success Score $>85\%$ | Deep alluvial/valley fill deposits, high TWI, structural intersections.
* **Interactive Inspector:** Clicking any grid cell triggers an API request returning the Borewell Success Probability Score ($0\text{--}100\%$), Recommended Drilling Depth ($\text{m}$), Topographic Wetness Index ($\text{TWI}$), Soil Permeability coefficient ($K$), and Distance to Structural Lineaments ($\text{m}$).

#### B. Managed Aquifer Recharge (MAR) Recommendations Panel
* **Objective:** Directs artificial recharge infrastructure placement based on surface permeability, urban density, and subsurface structure.
* **Layer Rendering:** Categorized Point Feature Markers coupled with Spatial Suitability Polygons.
* **Target Interventions:**
  * **Rooftop Rainwater Harvesting (RTRWH):** Flagged over high-density built-up zones where impervious concrete prevents natural percolation.
  * **Recharge Pits:** Recommended in low-slope agricultural depressions with high TWI and moderate soil permeability.
  * **Percolation Wells / Injection Shafts:** Placed precisely at intersections of fractured hard rock (high Lineament Density Index) to breach unsaturated vadose zones directly into deeper aquifers.

#### C. Groundwater Quality Intelligence Panel
* **Objective:** Maps spatial contamination trajectories and monitors municipal water health risks across the Berasia region.
* **Layer Rendering:** Inverse Distance Weighting (IDW) Interpolated Surface Heatmap layered with telemetry sample nodes.
* **Parameters Monitored:**
  * **Total Dissolved Solids (TDS):** Measured in $\text{mg/L}$.
  * **Fluoride ($\text{F}^-$) & Nitrate ($\text{NO}_3^-$):** Critical regional contaminants monitored against WHO/BIS thresholds.
  * **pH & Electrical Conductivity (EC):** Real-time sensor indicators of salinity and ionic concentration.
* **Alert Mechanism:** Calculating a Water Quality Index ($\text{WQI}$) $>100$ flags the node with a red alert, highlighting contaminated municipal sources for immediate containment.

#### D. Aquifer Health & Stress Assessment Panel
* **Objective:** Provides macro-level stress diagnostics combining Central Ground Water Board (CGWB) guidelines with dynamic ML head drawdown predictions.
* **Stress Classification Scheme:**
  $$\text{Stage of Extraction (SE)} = \left( \frac{\text{Annual Groundwater Draft}}{\text{Annual Rechargeable Resource}} \right) \times 100$$
  * **Safe (Green):** $\text{SE} < 70\%$ — Sustainable extraction baseline.
  * **Semi-Critical (Yellow):** $70\% \le \text{SE} < 90\%$ — Signs of seasonal water table decline.
  * **Critical (Orange):** $90\% \le \text{SE} \le 100\%$ — Steep drawdown; strict monitoring required.
  * **Over-Exploited (Red):** $\text{SE} > 100\%$ — Annual abstraction exceeds natural recharge.
* **Predictive Graphing:** Panel selection launches a 6-month predictive water table drawdown curve generated by the Physics-Informed Neural Network (PINN).

#### E. Dry Borewell Revival Intelligence Panel (Full-Scale Phase)
* **Status:** LOCKED / COMING SOON
* **Operational Mechanics:** Accepts user-submitted GPS coordinates of non-functional/failed borewells. Cross-references underlying lineament vectors, local hydro-geomorphology, and structural drainage patterns to output a Hydro-Fracturing Feasibility Score and Recharge Shaft Coupling Feasibility Index.

#### F. Digital Aquifer Twin Panel (Full-Scale Phase)
* **Status:** LOCKED / COMING SOON
* **Operational Mechanics:** Houses a Three.js / Deck.gl 3D WebGL engine rendering terrain surface elevation (DEM), unsaturated vadose zones, and bedrock geometry. Features interactive precipitation sliders to simulate real-time water table rise and fall based on physical continuity equations.

---

### 7.4 System Architecture & Tech-Stack Integration
![Image 28: Full-Stack Web-GIS Tech Stack Integration Schema](./images/image28.png)

---

### 7.5 End-to-End User Operational Workflow
![Image 29: End-to-End User Operational Workflow Diagram](./images/image29.png)
