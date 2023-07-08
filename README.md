Liver Segmentation Project with PyTorch and MONAI

This project aims to develop an efficient and accurate liver segmentation solution using the PyTorch deep learning framework and the MONAI (Medical Open Network for AI) library. The goal is to automatically identify and segment liver regions from medical images, such as CT scans or MRI data. 

Key Features:
- Utilizes PyTorch for building and training deep neural networks for liver segmentation.
- Integrates MONAI, a powerful open-source library for medical imaging, to streamline data preprocessing, augmentation, and evaluation.
- Implements state-of-the-art segmentation models, such as U-Net or DeepLab, tailored specifically for liver segmentation tasks.
- Provides pre-processing pipelines for handling medical image data, including loading, preprocessing, and data augmentation.
- Offers comprehensive evaluation metrics to assess the performance of the segmentation models, such as Dice Similarity Coefficient (DSC), Intersection over Union (IoU), and Hausdorff Distance.
- Supports GPU acceleration for faster training and inference on compatible hardware.

This repository contains the complete source code, pre-trained models, example datasets, and detailed documentation to facilitate liver segmentation research or applications in the medical field. Contributions and collaborations are welcome to further enhance the accuracy and robustness of the liver segmentation models.

Pytorch based patient liver screening ML Project :

![Output image](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/blob/main/images/liver_segmentation.PNG)


## Packages that need to be installed:
```
pip install monai
```
```
pip install -r requirements.txt
```
## Showing a patient from the dataset
How to present a patient: To address this, I created explicit scripts for how to show a patient from the training and testing datasets, which we can see here.

```Python
def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    """
    This function is to show one patient from your datasets, so that we can see if the it is okay or you need 
    to change/delete something.
    `data`: this parameter should take the patients from the data loader, which means you need to can the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `test`: this parameter is to say that you want to display a patient from the testing patients.
    """

    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    
    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_test_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_test_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

```

But before calling this function, we need to do the preprocess to your data, to visualize our patients after applying the different transforms so that we will know if we need to change some parameters or not.
The function that does the preprocess can be found in the `preprocess.py` file and in that file we will find the function `prepare()` that we can use for the preprocess.

## Training
After understanding how to do the preprocess we can start import the `3D Unet` from monai and defining the parameters of the model (dimensions, input channels, output channels...).

```Python
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
```

And to run the code, we can use the scripts `train.py`.

## Testing the model
To test the model, there is the jupyter notebook `testing.ipynb` file that contains the different codes that we need. We will be training/testing graphs about the loss and the dice coefficient and of course show the results of one of the test data to see the output of our model.

![Output image](https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch/blob/main/images/graphs.PNG)

----------------------------------------------------------------------------------------------------------------------------------


## Conversion tools



![154864750-c55a3129-67c7-438a-8549-e2c45c433048](https://user-images.githubusercontent.com/37108394/156251291-a0911b63-41b6-4c8a-820b-a9bfec5e452b.png)




