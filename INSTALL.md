## ⚙️ Installation Guide (Legacy)

> 🛑 **Note**: This project is no longer maintained. The following instructions are provided as a historical reference and may not work with current library versions or systems.

### 🐧 Setup on Linux (Ubuntu 20.04)

**Environment:**

* OS: Ubuntu Server 20.04
* Python: 3.8
* Package/Environment Manager: Miniconda

#### 1. Create virtual environment

```bash
conda create --name face3D python=3.8
conda activate face3D
```

#### 2. Install PyTorch3D (v0.7.2)

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
conda install -c bottler nvidiacub
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
```

#### 3. Install PyTorch (1.12.1 with CUDA 11.3)

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

#### 4. Install Python dependencies

You can either install them one by one or use the included `requirements.txt`.

```bash
pip install face-alignment==1.3.5
pip install opencv-python==4.4.0.46
pip install loguru==0.7.0
pip install matplotlib==3.3.4
pip install torchfile==0.1.0
pip install timm==0.6.13
pip install kornia==0.6.12
pip install chumpy==0.70
pip install numpy==1.19.5
pip install numba==0.55.0
pip install scikit-image==0.17.2
pip install cython==0.29.21
```

#### 5. Fix for PNG image loading error in `face-alignment`

Edit the following lines **only if you encounter issues with PNG images**:

* File: `face_alignment/detection/sfd/sfd_detector.py`, line 43
  Change:

  ```python
  image = self.tensor_or_path_to_ndarray(tensor_or_path)
  ```

  to:

  ```python
  image = self.tensor_or_path_to_ndarray(tensor_or_path)[:,:,:3]
  ```

* File: `face_alignment/api.py`, line 288
  Change:

  ```python
  image = io.imread(image_path)
  ```

  to:

  ```python
  image = io.imread(image_path)[:,:,:3]
  ```

---

### 🪟 Setup on Windows

#### 1. Create virtual environment (Python 3.8)

```bash
conda create -n 3Dface python=3.8
conda activate 3Dface
```

#### 2. Install PyTorch (1.12.1, CUDA 11.3)

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

#### 3. Install PyTorch3D

Official install guide: [https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

⚠️ **Important:** When building from source, ensure you activate the Conda environment *inside* the "x64 Native Tools Command Prompt for VS 2019" before running:

```bash
python setup.py install
```

#### 4. Additional required modules

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install pypiwin32
```

#### 5. Optional dependency fixes (Windows-only)

If you encounter errors with `win32con` or `win32file`, try the following:

* Edit:
  File: `.../site-packages/portalocker/portalocker.py`, line 8
  Replace:

  ```python
  import win32con
  ```

  with:

  ```python
  import win32.lib.win32con as win32con
  ```

* Then install:

  ```bash
  conda install pywin32
  ```

📌 Reference: [https://stackoverflow.com/questions/58612306/how-to-fix-importerror-dll-load-failed-while-importing-win32api](https://stackoverflow.com/questions/58612306/how-to-fix-importerror-dll-load-failed-while-importing-win32api)

---

### 🧩 Blender Add-on Integration

To visualize or reconstruct using Blender:

#### 1. Install Blender add-on

* Locate the add-on file:
  `Source/networks/addon_Blender_API_Face_Reconstruction_v2.py`
* In Blender:

  ```
  Edit > Preferences > Add-ons > Install > Select the file above > Install Add-on
  ```

#### 2. Run model inference server

```bash
cd networks
python APIBlender_v2.py
```

Use Blender demo scenes as reference.

---