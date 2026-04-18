# 🌊 Learning Mesh-Based Simulation with Graph Networks
### *Fast, Adaptive, and Physics-Informed Neural Simulators for Complex Fluid Dynamics*

This repository provides a **PyTorch + PyG (PyTorch Geometric)** implementation of **MeshGraphNets**—a powerful graph neural network framework for learning mesh-based physical simulations. We focus on the **flow around a circular cylinder** problem, reproducing and extending the groundbreaking work from DeepMind.

> 🔬 **Original Paper**:  
> [**Learning Mesh-Based Simulation with Graph Networks**](https://arxiv.org/abs/2010.03409)  
> *Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia*  
> **ICLR 2021**

---

## ✨ Why This Project?

- **Physics-aware learning**: Leverages mesh structure to respect geometric and physical priors.
- **High performance**: Runs **10–100× faster** than traditional solvers while maintaining fidelity.
- **Extensible**: Built on PyTorch Geometric—easy to adapt to new PDEs, materials, or domains.

---

## 🛠️ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

> 💡 **Note**: TensorFlow < 1.15.0 is required only for parsing the original TFRecord datasets.

---

## 🚀 Quick Start

### 1. Download the Dataset

We use DeepMind's `cylinder_flow` dataset:

```bash
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -d data
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord -d data
```

### 2. Parse TFRecords

Convert to PyTorch-friendly format:

```bash
python parse_tfrecord.py
```

> Output saved in `./data/`.

### 3. Train the Model

```bash
python train.py
```

For multi-GPU training:

```bash
export NGPUS=2  # set as your machine's available GPUs
torchrun --nproc_per_node=$NGPUS train_ddp.py --dataset_dir data
```

### 4. Run Rollouts & Visualize

Generate long-horizon predictions and render videos:

```bash
python rollout.py          # saves results to ./results/
python render_results.py   # generates videos in ./videos/
```

### 5. Export to ONNX (Optional)

Export model for visualization and deployment:

```bash
# Export with random weights (architecture test)
python export_onnx.py --output model.onnx

# Export with trained weights
python export_onnx.py --checkpoint checkpoints/best_model.pth --output model.onnx --visualize

# Visualize in Netron
pip install netron
netron model.onnx
```

> 📊 See detailed guide: [`docs/onnx-export-guide.md`](docs/onnx-export-guide.md)

---

## 🎥 Demos

### Results on DeepMind's `cylinder_flow`:

| Demo 0 | Demo 1 |
|------------|--------------|
| ![Demo 0](videos/0.gif) | ![Demo 1](videos/1.gif) |

### Results on **our own CFD-generated data** (new geometries & conditions):

| Demo 2 | Demo 3 |
|------------|--------------|
| ![Demo 2](videos/2.gif) | ![Demo 3](videos/3.gif) |

> ✅ The model generalizes well—even to unseen flow regimes and mesh configurations!

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/understanding-notes.md`](docs/understanding-notes.md) | Comprehensive code architecture and training pipeline notes |
| [`docs/data-structure-explained.md`](docs/data-structure-explained.md) | Detailed explanation of data structures and model consumption flow |
| [`docs/dataset-analysis.md`](docs/dataset-analysis.md) | In-depth analysis of the cylinder_flow dataset |
| [`docs/onnx-export-guide.md`](docs/onnx-export-guide.md) | Guide for exporting model to ONNX format |

---

## 🔍 Key Features

### Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MeshGraphNet                           │
├─────────────────────────────────────────────────────────────┤
│  Encoder    →  Map node/edge features to 128D hidden space  │
│      ↓                                                      │
│  GnBlock ×15 →  Message passing (edge + node updates)       │
│      ↓                                                      │
│  Decoder    →  Output acceleration prediction (2D)          │
└─────────────────────────────────────────────────────────────┘
```

### What the Model Predicts

- **Output**: Acceleration field `a(t) = dv/dt` at each time step
- **Integration**: Velocity updated via `v(t+1) = v(t) + a(t)`
- **Why acceleration?**: More stable numerically, physically meaningful (F=ma)

### Input Features

| Feature | Shape | Description |
|---------|-------|-------------|
| `node_attr` | `[N, 11]` | Node features (velocity 2D + node type one-hot 9D) |
| `edge_attr` | `[E, 3]` | Edge features (dx, dy, distance) |
| `edge_index` | `[2, E]` | Graph connectivity |

---

## 🧪 Testing & Inference

### Single-step inference

```bash
python demo_inference.py --checkpoint checkpoints/best_model.pth
```

### Multi-step rollout with visualization

```bash
python rollout.py --checkpoint checkpoints/best_model.pth --num_steps 600
python render_results.py
```

---

## 📊 Dataset Statistics

| Split | Trajectories | Nodes | Cells | Time Steps | Size |
|-------|--------------|-------|-------|------------|------|
| Train | 1,000 | ~1,876 | ~3,518 | 600 | 12.71 GB |
| Valid | 100 | ~1,896 | ~3,558 | 600 | 1.27 GB |
| Test  | 100 | ~1,923 | ~3,612 | 600 | 1.26 GB |

**Node Types**: NORMAL (87.5%), WALL (10.5%), INFLOW (0.9%), OUTFLOW (0.9%)

---

## 📬 Contact

Have questions, suggestions, or want to collaborate?  
📧 Reach out: [jianglx@whu.edu.cn](mailto:jianglx@whu.edu.cn)

---

> ⭐ **If you find this project useful, please consider starring the repo!**  
> Your support helps us keep improving open-source scientific ML tools.
