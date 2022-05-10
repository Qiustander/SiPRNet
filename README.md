# SiPRNet: End-to-End Learning for Single-Shot Phase Retrieval
#### Members: Qiuliang Ye, Li-Wen Wang, Daniel P. K. Lun

![](figures/architecture.png)


## Prerequisites
- Linux
- Python 3
- NVIDIA GPU  + CUDA cuDNN

## Testing
- A experimental testing images are included in the `Data` folder.
- The trained model for the RAF dataset can be found in `Model/SiPRNet_RAF.pth`.
- The trained model for the Fashion-MNIST dataset can be found in `Model/SiPRNet_Fashion.pth`.
- Test the model:
```bash
python Save_result_SiPRNet.py
```
The test results will be saved to the folder: `./Result`.

#### Please change `dataset_name` (RAF/Fashion) in the `Save_result_SiPRNet.py` to generate the results with specific datasets.


