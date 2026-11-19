# Auto-Encoder Project

Class project from **Day 4 of the Intel NEC Deep Learning Workshop**. It covers plain, denoising and variational autoencoders on MNIST, then applies them to real problems: weather anomaly detection, credit-card fraud detection and a deployed Gradio app.

**Live demo:** https://0xarchit-auto-encoder.hf.space  
**Hugging Face Space:** https://huggingface.co/spaces/0xarchit/auto-encoder

## What's inside

```
Auto-Encoder-Project/
├── m4-autoencoders-2415800016/      # Notebook 1: the lab
│   ├── m4_autoencoders_2415800016.ipynb
│   ├── dae_2415800016.pt            # trained denoising AE weights
│   └── journal_2415800016.json      # predictions, results, explanations
├── m4-ae-project-2415800016/        # Notebook 2: the project series
│   ├── m4_ae_project_2415800016.ipynb
│   ├── journal_P_2415800016.json
│   ├── vectors_2415800016.tsv       # latent codes for TF Embedding Projector
│   ├── metadata_2415800016.tsv      # labels for the projector
│   ├── feedback_2415800016.csv      # feedback collected from the app
│   └── p1_patch_2415800016.patch
└── hf/                              # Hugging Face Space (Gradio app)
    ├── app.py
    ├── requirements.txt
    ├── dae_2415800016.pt
    └── README.md
```

## Notebook 1: Autoencoder lab (MNIST)

Model: `784 → 256 → 64 → z → 64 → 256 → 784` (ReLU, Sigmoid output), latent size `z = 3`, 435,987 parameters, trained with MSE and Adam.

| Experiment | Result |
|---|---|
| Reconstruction (z=3, 5 epochs) | Test MSE **0.0378**. Digit 1 is easiest (0.0105), digit 5 hardest (0.0492) |
| Bottleneck sweep | z=2: 0.0482, z=3: 0.0407, z=16: 0.0258, z=64: 0.0244. Gains flatten after z≈16 |
| Latent space (z=2) | Digits 4/9 and 3/5/8 overlap in the 2-D plot, with no labels used in training |
| Denoising AE (σ=0.45) | PSNR: noisy input 10.11 dB, plain AE 12.37 dB, **denoising AE 13.94 dB** |
| Anomaly detection | Trained without digit 6; reconstruction error as score gives **ROC-AUC 0.895** |
| VAE (β sweep) | β=0.25: MSE 0.0388 / KL 11.07; β=1: 0.0395 / 7.16; β=4: 0.0427 / 4.44. Higher β gives lower KL and blurrier output |
| Morphing | Latent-space walk between 7 and 0 gives sharp, realistic intermediates, unlike a plain pixel blend |

## Notebook 2: Project series

- **P1: Hacking the official PyTorch VAE.** Read `pytorch/examples/vae`, changed the latent size by hand, added a `--beta` argument (`BCE + β·KLD`) and ran failure cases (β=0 and β=20: posterior collapse).
- **P2: Stable Diffusion's VAE.** Tested `stabilityai/sd-vae-ft-mse` on personal photos. A 512×512×3 image becomes a 64×64×4 latent, a **48× compression**. PSNR: face 28.41, text 27.86, crowd 24.33, texture 17.89. Small text and distant faces fail worst. Also tried latent surgery (scaling, noise, blending).
- **P3: Weather anomaly detector (Chennai).** Open-Meteo data 1991–2024, 10-day windows, trained on ≤2014 with monthly standardisation. The top anomaly was **9–18 Dec 2016 (Cyclone Vardah)**; 8 of the top 10 windows matched real events. A live check for 19 Nov 2026 scored 0.549 against a threshold of 1.278, so **normal**.
- **P4: Credit-card fraud on a budget.** 284,807 transactions, 0.17% fraud, 100-alert review budget:

  | Model | Precision | Recall | Net saving |
  |---|---|---|---|
  | Isolation Forest (baseline) | 0.02 | 0.027 | -€465 |
  | AE `[16, 8]`, 5 epochs | 0.12 | 0.16 | -€303 |
  | AE `[20, 10, 4]`, 10 epochs | 0.25 | 0.333 | -€152 |
  | **Denoising AE `[32, 12]`, 12 epochs, noise 0.05** | **0.50** | **0.667** | **+€4,680** |

- **P5: Latent space in 3-D.** Exported latent codes to the [TensorFlow Embedding Projector](https://projector.tensorflow.org). t-SNE and UMAP separate fraud better than PCA; frauds form scattered outlier clusters rather than one tight group.
- **P6: Deployed app.** Gradio app on Hugging Face Spaces (see below).

## The Gradio app (`hf/`)

Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/0xarchit/auto-encoder). The `hf/` folder is the same code that runs there.

1. **Denoiser:** upload a digit photo or pick an MNIST test image, add noise, see clean / noisy / denoised side by side.
2. **Weather alarm:** pulls live data from Open-Meteo and scores the last 10 days with the weather autoencoder.
3. **Feedback:** collects ratings and comments into a CSV.

Run locally:

```bash
cd hf
pip install -r requirements.txt
python app.py
```

Note: `app.py` imports `spaces` and uses `@spaces.GPU` for Hugging Face ZeroGPU. To run on your own machine, remove those lines and use `cuda`/`cpu` as available.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, pandas, matplotlib, Gradio, Hugging Face (`diffusers`, Spaces), Open-Meteo API, OpenML.

## Credits

- MNIST handwritten digits
- Credit-card fraud dataset by ULB / Worldline (OpenML)
- Weather data by [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0)
- Reference VAE from [pytorch/examples](https://github.com/pytorch/examples)

## Author

**Archit** ([@0xarchit](https://github.com/0xarchit))
