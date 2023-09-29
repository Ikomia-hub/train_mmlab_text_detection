<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_mmlab_text_detection/main/icons/mmlab.png" alt="Algorithm icon">
  <h1 align="center">train_mmlab_text_detection</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_mmlab_text_detection">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_mmlab_text_detection">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_mmlab_text_detection/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_mmlab_text_detection.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train text detection models from MMLAB.

![example](https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_text_detection/main/icons/results.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

To try this code snippet, you can download and extract from [wildreceipt](https://download.openmmlab.com/mmocr/data/wildreceipt.tar).
Then make sure you fill the parameter **dataset_folder** correctly.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add text recognition dataset
dataset = wf.add_task(name="dataset_wildreceipt", auto_connect=False)

# Set dataset parameters
dataset.set_parameters({'dataset_folder': "/path/to/dataset/folder"})

# Add train algorithm
train = wf.add_task(name="train_mmlab_text_detection", auto_connect=True)

# Set train algorithm parameters
train.set_parameters({'model_name': 'dbnetpp', 
                      'cfg': 'dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015',
                      'epochs': '10',
                      'batch_size': '2'})

# Launch training
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str, default="dbnet"): name of pretrained model.
- **cfg** (str, default="dbnet_resnet18_fpnc_1200e_icdar2015.py"): filename of pretrained model's config.  

**model_name** and **cfg** work by pair. You can print the available possibilities with this code snippet:
```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="train_mmlab_text_detection")

# Get model zoo and print it
model_zoo = algo.get_model_zoo()
print(model_zoo)

# Set parameters with the first model of the list
algo.set_parameters(model_zoo[0])
```

- **epochs** (int, default=10): number of complete passes through the training dataset.
- **batch_size** (int, default=4): number of samples processed before the model is updated.
- **dataset_split_ratio** (int, default=90): in percentage, divide the dataset into train and evaluation sets ]0, 100[.
- **output_folder** (str): path to where the model will be saved. Default folder is "runs/" in the algorithm directory.
- **eval_period** (int, default=1): interval between evaluations.
- **dataset_folder** (str): path to where the dataset compatible with mmlab is stored. Default folder is "/dataset" in the algorithm directory.
- **expert_mode** (bool, default=False): set to True only if you know how mmlab works. Then you can set all the parameters in the mmlab config system and it will override every other parameters above.
- **config_file** (str, default=""): path to the .py config file. Only for custom models.
- **model_weight_file** (str, default=""): path to the .pth weight file. Only for custom models.  

*Note*: parameter key and value should be in **string format** when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="train_mmlab_text_detection", auto_connect=True)

algo.set_parameters({
    "model_name": "dbnetpp",
    "cfg": "dbnetpp_resnet50_fpnc_1200e_icdar2015.py",
    "epochs": "20",
    "batch_size": "2",
    "eval_period": "2",
    "dataset_split_ratio": "90",
    "output_folder": "/out",
    "dataset_folder": "/dataset",
    "export_mode": "False",
    "config_file": "",
    "model_weight_file": ""
})

# Continue your workflow
```

## :fast_forward: Advanced usage 

[optional]
