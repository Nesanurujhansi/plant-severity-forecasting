# Project Glossary

This document provides plain-language definitions for key technical terms used across machine learning, computer vision, plant pathology, statistics, and web software engineering in **crop-prognosis**.

---

## Machine Learning & Computer Vision Terms

1. **Backbone**: A pre-trained neural network (such as DINOv2) that converts raw pixels into rich mathematical feature representations.
2. **Feature**: A numerical representation of an image or dataset attribute used by machine learning models to make decisions.
3. **Feature Embedding**: A dense vector of numbers summarizing visual traits like texture, shape, and color extracted by a backbone.
4. **Frozen Model**: A neural network whose weights are locked during training so that its learned feature representation remains fixed.
5. **Head**: A small neural network attached to the output of a backbone to solve a specific downstream task (e.g. segmentation or classification).
6. **Segmentation**: The process of classifying every individual pixel in an image as belonging to a specific category (e.g. leaf, lesion, background).
7. **Mask**: A binary matrix (0s and 1s) representing the exact pixel location of an object in an image.
8. **Lesion**: A visible spot, blotch, or damaged tissue on a plant leaf caused by fungal or bacterial disease.
9. **IoU (Intersection over Union)**: A metric measuring segmentation accuracy calculated as overlapping pixels divided by total combined pixels.
10. **Dice Coefficient**: A similarity metric for image segmentation defined as twice the overlapping area divided by the sum of both areas.
11. **OOD (Out-of-Distribution)**: Data inputs that differ significantly from the dataset used during model training (e.g., an image of a shoe uploaded to a leaf app).
12. **Vision Transformer (ViT)**: A deep learning vision architecture that processes images by dividing them into patches and applying self-attention mechanisms.
13. **Pseudo-labeling**: Using an automated model (like SAM 2) to generate initial training annotations when human labels are unavailable.
14. **Connected Components**: An image processing technique that groups neighboring pixels to count distinct disease spots and measure their size statistics.
15. **Union Rule**: An invariant rule ($\text{leaf} = \text{sam2\_mask} \cup \text{lesion\_mask}$) guaranteeing that lesion pixels are strictly contained inside the leaf boundary.
16. **Segment Anything Model 2 (SAM 2)**: A foundation vision segmentation model by Meta AI capable of zero-shot object boundary prediction.
17. **Perceptual Hash (`dhash`)**: A visual fingerprint algorithm converting image pixel gradients into a 64-bit integer to detect duplicate or near-duplicate photos.
18. **Hamming Distance**: The number of bit positions in which two binary hash strings differ, used to cluster near-duplicate images ($\le 10$).
19. **Contact Sheet**: A visual quality grid displaying multiple sample images with translucent mask overlays to inspect label quality.

---

## Epidemiology & Statistical Terms

20. **Severity**: The proportion of a leaf's total surface area damaged by disease, expressed as a fraction (0.0 to 1.0) or percentage (0% to 100%).
21. **Incidence**: The proportion of leaves or plants affected by a disease in a given plot or population.
22. **Calibration**: The degree to which predicted probabilities or uncertainty intervals match true real-world frequencies.
23. **Conformal Prediction**: A statistical technique that guarantees exact prediction interval bounds without assuming a specific data distribution.
24. **Prediction Interval**: A numerical range (e.g., 10% to 15%) that contains the true severity value with a specified probability (e.g., 90%).
25. **Mondrian Conformal Prediction**: Grouping conformal calibration data into bins (e.g. severity classes) so each bin receives its own tailored uncertainty bound.
26. **Coverage**: The empirical proportion of test set targets that fall within the predicted conformal intervals.
27. **Logistic Growth**: S-shaped biological growth curve where rapid early growth slows down as resources or space become scarce.
28. **Thermal Response Curve \(g_T(T)\)**: A bounded biological mathematical function describing how pathogen growth rate varies with ambient temperature.
29. **Humidity Response Curve \(g_H(RH)\)**: A function describing how relative humidity influences fungal spore germination and lesion expansion.
30. **Residual**: The numerical difference between an actual observed value and the baseline biological growth model prediction.
31. **LightGBM**: A fast, gradient-boosted decision tree library optimized for tabular data predictions.
32. **Quantile**: A cutoff value dividing a probability distribution into equal probability intervals (e.g. 10th, 50th, 90th percentiles).
33. **CQR (Conformalized Quantile Regression)**: A method combining quantile regression with conformal prediction to form valid predictive intervals.
34. **Data Leakage**: An error where test or calibration information improperly enters the training process, producing artificially inflated performance metrics.
35. **Cross-Validation**: A evaluation technique partitioning dataset rows into folds to evaluate model generalization on unseen data.
36. **Fold**: A single train/validation data split used in cross-validation experiments.
37. **Mixed-Effects Model**: A statistical model accounting for both fixed population effects (e.g. fungicide treatment) and random grouping effects (e.g. cultivar or plot).
38. **Cluster Bootstrap**: Resampling whole clusters (e.g., plots or leaf time series) with replacement to compute reliable confidence intervals.
39. **Lin's Concordance Correlation Coefficient (CCC)**: A metric measuring both accuracy and precision of continuous predictions against ground truth (1.0 = perfect agreement).
40. **Bland-Altman Plot**: A statistical plot showing the mean difference (bias) and 95% limits of agreement ($\pm 1.96 \cdot \text{SD}$) between two measurement methods.
41. **Quadratic Weighted Kappa (QWK)**: A statistical metric measuring agreement between predicted and true ordinal classes, penalizing larger class distance errors quadratically.
42. **Ordinal Severity Class**: Categorical disease severity bins ($<1\%$, $1-5\%$, $5-10\%$, $10-25\%$, $25-50\%$, $>50\%$) grouping continuous severity fractions into discrete severity levels.
43. **Skill Score vs Persistence**: A normalized forecast evaluation metric ($1 - \text{MAE}_{\text{model}} / \text{MAE}_{\text{persistence}}$) measuring improvement over assuming tomorrow's severity equals today's.
44. **Feature Parity**: Guaranteeing that feature vectors computed during offline training on tabular datasets match those built during online inference down to $10^{-6}$ precision.
45. **PINN-ODE (Physics-Informed Neural ODE)**: A hybrid differential equation combining biological growth curves with neural parameter functions to enforce physical conservation principles.
46. **Yan & Hunt Beta Curve**: A non-symmetric biological thermal response function modeling pathogen growth across cardinal temperature bounds ($T_{\text{min}}, T_{\text{opt}}, T_{\text{max}}$).
47. **Vapor Pressure Deficit (VPD in kPa)**: A thermodynamic measure of the drying capacity of air, calculated via the Magnus formula to quantify fungal spore germination potential.
48. **TarNet (Deep Counterfactual Treatment Network)**: A causal neural network architecture featuring a shared representation trunk and dual treatment branch heads to estimate potential outcome trajectories.
49. **Free Energy OOD Score**: An Out-of-Distribution scoring metric ($E(\mathbf{x}) = -T \cdot \log \sum \exp(f_i/T)$) derived from the unnormalized log partition function of feature representations.

---

## Software, Storage & System Terms

47. **Manifest**: A structured metadata index file recording checksums, file paths, and creation timestamps for dataset splits.
48. **Google Colab**: A cloud-based Python notebook platform providing temporary access to GPU accelerators for training models.
49. **FastAPI**: A high-performance Python web framework for building asynchronous HTTP REST APIs.
50. **Gradio**: A Python framework for rapidly building web user interfaces for machine learning applications.
51. **Endpoint**: A specific URL path (such as `/health` or `/predict`) where a server listens for incoming HTTP client requests.
52. **Schema**: A formal definition of input or output data structures validated using Pydantic.
53. **Abstain**: The decision by the safety system to refuse making a prediction when an input photo is blurry, dark, or out-of-distribution.
54. **Feature Parity**: Ensuring that feature calculation logic in production serving code perfectly matches the logic used during model training.
55. **Shrinkage Covariance**: A covariance regularization technique blending sample covariance with an identity scaling matrix to ensure stable matrix inversion.
56. **Mahalanobis Distance**: A distance metric measuring how many standard deviations a sample feature vector is from the mean of a multivariate distribution.
57. **CLS Token**: A special summary token output by a Vision Transformer that represents global image-level context.
58. **Patch Tokens**: Spatial feature vectors corresponding to small square tiles (e.g. 14x14 pixels) in an image.
59. **Storage Footprint Estimator**: A tool calculating exact disk space requirements for feature caches prior to extraction.
60. **Laplacian Variance**: An image sharpness metric calculating the variance of the 2D discrete Laplacian operator across grayscale pixels.
61. **Module Gating**: A safety policy mechanism that selectively enables or disables specific downstream application modules (e.g. severity, progression, scenario) based on input validity.
62. **Confidence Tier**: A safety rating (`HIGH`, `MEDIUM`, `LOW`) assigned to an inference request indicating the reliability of photo quality and distribution fit.
63. **Zero-Pickle Serialization**: Storing model weights and parameters exclusively in safe formats (`.safetensors`, JSON, text) to prevent remote code execution vulnerabilities.
64. **SHA-256 Checksum**: A 64-character cryptographic hash value acting as a unique digital fingerprint to detect file corruption or modification.
65. **Model Bundle**: A zip archive containing all model weight files, configs, calibration metrics, manifest index, and golden reference tests required for web app serving.
66. **Golden Reference Test**: A benchmark test using fixed synthetic reference inputs and expected prediction outputs to verify local CPU inference accuracy against GPU reference outputs.
67. **Zip-Slip**: A security vulnerability where malicious zip archives contain path traversal strings (`../`) attempting to overwrite system files outside the extraction directory.
68. **Development Stub Mode (`DEV_STUB_MODELS`)**: A software fallback flag enabling the web app to run with lightweight mock predictors when real model weight files are not available.
69. **Decompression Bomb Guard**: A security check limiting the maximum pixel count ($width \times height \le 25\text{ MP}$) of an uploaded image to prevent memory exhaustion attacks.
70. **EXIF Stripping**: Removing raw exchangeable image file format metadata (such as GPS coordinates, camera serial numbers, and capture timestamps) from uploaded photos prior to processing to preserve user privacy.
71. **Weather Scenario Presets**: Standardized future weather assumption profiles (`as_recent`, `climatology`, `warm_humid`, `cool_dry`, `user_entered`) used to project disease growth trajectories without making false meteorological forecast claims.
72. **Inference Pipeline**: A master orchestrator class executing image validation, safety checks, severity prediction, microclimate resolution, progression forecasting, and treatment scenario simulation in sequential order.
73. **Request Tracing ID (`X-Request-ID`)**: A unique UUID assigned to each HTTP request header allowing developers to trace processing logs across server modules without logging sensitive payload bytes.

---

## Farmer-Friendly Plain-Language Equivalents

To ensure non-academic and non-technical agricultural users can immediately understand their crop health reports, all technical terms in user-facing UI and PDF reports map to plain-language equivalents:

| Technical / Scientific Term | Farmer-Friendly Plain-Language Equivalent | Meaning & Context |
| :--- | :--- | :--- |
| **Foliar** | **Leaf** | Pertaining to plant leaves. |
| **Necrotic Core** | **Dead Brown Spots** | Dead, dry brown or black center tissue killed by pathogen. |
| **Chlorotic Halo / Margin** | **Spreading Yellow Border** | Active yellowing edge where disease is currently invading green tissue. |
| **Two-Tier Pathology Mask** | **Disease Spot Outlines** | Visual outlines distinguishing dead spots from spreading yellow edges. |
| **Foliar Envelope** | **Healthy Leaf Area / Outline** | Green vegetative leaf boundary excluding background. |
| **Attention Saliency** | **AI Focus Areas** | Visual heatmap showing the exact leaf areas examined by the AI vision model. |
| **NWP Microclimate** | **High-Accuracy Local Weather** | 9 km numerical weather forecasts for temperature, humidity, and rain. |
| **Conformal Prediction Band** | **Expected Damage Range** | Mathematically guaranteed 90% confidence range for leaf damage. |
| **Epidemiological Forecast** | **Forecasted Disease Spread** | Predicted multi-day expansion of disease spots if left untreated. |
| **Active Lesion Spots** | **Disease Spots Found** | Number of distinct connected disease spots counted on the leaf. |
| **Biological Foliar Application** | **Safe Leaf Spraying** | Applying approved organic/biological liquids during calm weather windows. |


