### 1. DICOM/NIfTI Extraction & HU Windowing
Raw CT scans are not standard images (like JPEGs); they are arrays of physical measurements called **Hounsfield Units (HU)**, which typically range from -1000 (air) to +3000 (dense bone).
* **The Operation:** You apply a specific "Lung Window" filter.
* **The Goal:** You clip the pixel values to isolate only the lung tissue and soft tissues, ignoring the bones and the background air outside the body. This usually means restricting the HU values to a range like -1250 to +250. This explicitly forces the AI to focus on lung structures (like GGOs) rather than getting distracted by the ribs.

### 2. Normalization (Min-Max Scaling)
Neural networks hate large, erratic numbers. They train best when input values are small and consistent.
* **The Operation:** You convert the restricted HU values into floating-point numbers.
* **The Math:** You scale every pixel so the entire image falls strictly into a range of **0.0 to 1.0** (or **-1.0 to 1.0**, depending on your specific GAN setup).
* **The Goal:** This stabilizes the gradient descent during training, preventing the networks (especially the SRGAN) from experiencing vanishing or exploding gradients.

### 3. Spatial Standardization (Resizing & Padding)
Every patient's body size is different, and hospital CT scanners can output slightly different dimensions.
* **The Operation:** You resize or crop/pad all slices to a strict, uniform spatial dimension.
* **The Goal:** Deep learning models require fixed input sizes. In your pipeline, this usually means forcing every single 2D slice to exactly **512 x 512** pixels (or 256 x 256 if you are heavily constrained by GPU memory).

### 4. The 2.5D Tensor Construction (The Core Innovation)
This is the most critical preprocessing step unique to your specific framework. Once the images are windowed, normalized, and resized, you do not feed them into the network one by one.
* **The Operation:** For every target slice $i$ that you want to process, the data loader automatically grabs slice $i-2$, slice $i-1$, slice $i+1$, and slice $i+2$.
* **The Concatenation:** It stacks these 5 separate grayscale 2D slices along the depth/channel axis.
* **The Final Output:** The data loader spits out a single $5 \times H \times W$ tensor.
* **The Handoff:** This perfectly prepared $5 \times 512 \times 512$ tensor now exits the preprocessing stage and is fed directly into the first convolutional layer of **Block 2 (The DnCNN)** to begin the noise restoration process.