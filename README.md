# Building Extraction

## Model Selection, Adaptation Strategy, and Evaluation

### Problem Framing and Dataset Analysis

The task of building extraction from aerial imagery involves identifying and segmenting individual building footprints from complex urban scenes.  
An exploratory analysis of the official dataset revealed that buildings vary significantly in **shape, orientation, and density**, with frequent **very close rooftops** and occasional **illumination or shadow ambiguity** that made the separation between structures challenging.

To address these challenges, the central goal of this work was to develop a segmentation model capable of **capturing global contextual information** while remaining sensitive to **fine local details** — a balance that is often difficult to achieve with conventional architectures.

---

### Architecture Choice

Recent studies such as [Zhang et al., 2024](https://www.sciencedirect.com/science/article/pii/S1077314224003345) and [Zhou et al., 2024](https://dl.acm.org/doi/10.1145/3696474.3696490) have shown that **Transformer-based architectures** are increasingly effective for **building and urban scene segmentation**.  
They outperform purely convolutional approaches in terms of spatial consistency and generalization, thanks to their ability to integrate **global attention** across the entire image.

The **SegFormer** architecture was selected because it combines **hierarchical transformers** with a lightweight segmentation head, achieving an excellent trade-off between **accuracy, model size, and inference speed**.  
Its encoder captures both **fine-grained local information** and **global context**, making it especially suitable for:
- distinguishing **adjacent or overlapping rooftops**,  
- adapting to **different urban textures and lighting conditions** and  
- generalizing across **multiple cities or regions**.

The chosen **SegFormer-B0** variant was originally **pre-trained on ADE20K**, a dataset containing numerous **urban and architectural scenes**.  
This pre-training offered a useful inductive bias for recognizing **man-made structures**, which accelerated convergence and improved feature transfer to the building extraction task.

---

### Domain Adaptation with OpenStreetMap (OSM)

To further improve generalization, a **domain adaptation strategy** was implemented using **OpenStreetMap (OSM)** data.  
The OSM dataset provided additional imagery and segmentation masks from **Kyoto, Osaka, Sapporo, and Fukuoka**, four Japanese cities with distinct **architectural layouts, densities, and roof typologies**.  
This extended dataset diversified the training distribution, enabling the model to better capture the visual variability found in urban environments.

Two adaptation strategies were explored:

1. **`segformer_finetuned.pth` — Sequential Domain Adaptation**  
   The model was **pre-trained for 5 epochs on OSM data** and then **fine-tuned exclusively on the official dataset**.  
   This sequential approach allowed the model to first learn **generic structural patterns** of buildings from OSM before specializing to the competition’s higher-quality annotations.  
   The two-stage setup encouraged more stable convergence and stronger domain alignment.

2. **`segformer_full_finetune_osm_mix.pth` — Joint Domain Adaptation**  
   In this configuration, the model was fine-tuned on a **merged dataset** that combined OSM and official data in a single training phase.  
   This exposed the model to greater variability during learning but occasionally introduced **annotation inconsistencies** between domains.

---

### Evaluation Metrics and Methodology

Performance was assessed using both **pixel-wise** and **object-wise** metrics to evaluate segmentation quality comprehensively:

- **Pixel-wise IoU** and **F1-score**, to assess segmentation overlap quality;  
- **Object-wise Precision**, **Recall** and **F1@0.5 IoU**, to assess instance-level accuracy by matching predicted polygons to ground truth with an IoU threshold of 0.5.

These metrics were selected because they capture different aspects of model performance:  
pixel-wise measures reflect local accuracy, while object-wise metrics penalize **under-segmentation** (merged buildings) and **over-segmentation** (fragmented roofs).

---

### Post-processing and Instance Refinement

The segmentation outputs often contained **merged rooftops** or **incomplete contours**, especially in areas with very close buildings or inconsistent illumination.  
To address this, a **morphological post-processing** step was introduced:

1. **Erosion** — separates adjacent buildings by shrinking connected regions;  
2. **Connected-component labeling** — isolates individual structures;  
3. **Dilation and contour simplification** — restores boundaries and converts masks into polygonal outputs.

The parameters controlling these operations (erosion/dilation iterations, threshold, and polygon simplification) were chosen through a **grid search (parameter sweep)** on the validation set to balance **building separation** and **shape preservation**.  

This post-processing (denoted as **Morph + CC Split**) improved **instance separation** and ensured more geometrically consistent predictions while maintaining low inference cost.

However, some errors persisted in regions where **buildings were extremely close** or **visual gaps were imperceptible** due to lighting or terrain similarity.  
In such cases, both models tended to **merge multiple buildings into one instance**, a limitation that could be addressed with future instance-aware architectures.

---

### Quantitative Results

#### Pixel-wise Evaluation

| Model | IoU (pixel) | F1 (pixel) |
|--------|--------------|-------------|
| `segformer_finetuned.pth` | **0.786** | **0.874** |
| `segformer_full_finetune_osm_mix.pth` | 0.780 | 0.869 |

#### Object-wise Evaluation (with Morph + CC Split)

| Model | F1@0.5 IoU | Precision | Recall | Avg. Time/Image |
|--------|-------------|------------|---------|----------------|
| `segformer_finetuned.pth` | **0.496** | **0.640** | 0.433 | 0.0467 s |
| `segformer_full_finetune_osm_mix.pth` | 0.489 | 0.622 | 0.431 | 0.0461 s |

Both models achieved high pixel-level consistency and competitive object-level detection.  
The **sequential domain adaptation** (`segformer_finetuned.pth`) showed slightly superior overall F1 and precision, suggesting more stable generalization and cleaner segmentation boundaries.

---

### Conclusions

- The **SegFormer** architecture, based on **hierarchical transformers**, effectively balances global context and fine-grained detail, enabling accurate building segmentation in diverse urban scenarios.  
- **Pre-training on OSM data** from multiple Japanese cities improved robustness to architectural variability and reduced domain bias.  
- The **sequential domain adaptation** approach outperformed the joint setup, achieving higher precision and generalization stability.  
- **Morphological post-processing** significantly improved instance separation, although fine-grained adjacency errors still occurred under difficult lighting or terrain conditions.

The final model, **`segformer_finetuned.pth`**, achieved the best trade-off between **accuracy, robustness, and inference speed**, making it suitable for large-scale building extraction pipelines.

---

### Future Work

Several directions could further enhance performance:

- **Instance-aware segmentation models**, such as **Mask R-CNN** could explicitly handle overlapping and touching buildings, though at higher computational cost.  
- **Semi-supervised or self-training strategies** could exploit large volumes of unlabeled OSM imagery to further boost generalization.

These extensions would push the model toward a fully scalable, domain-agnostic system for automatic urban structure mapping.

---
