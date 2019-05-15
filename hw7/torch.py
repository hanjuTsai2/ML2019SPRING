{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "import utils\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_folder = 'data/images/'\n",
    "data_dirs = os.listdir(data_folder)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "test = utils.loadImage(os.path.join(data_folder, data_dirs[9]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir_num = [ int(i.split('.jpg')[0]) for i in data_dirs]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import os\n",
    "from PIL import Image\n",
    "from torch.utils.data.dataset import Dataset\n",
    "import numpy as np\n",
    "\n",
    "class DatasetProcessing(Dataset):\n",
    "    def __init__(self, data_path, img_path, transform=None):\n",
    "        self.img_path = os.path.join(data_path, img_path)\n",
    "        self.transform = transform\n",
    "        self.img_filename = os.listdir(self.img_path)\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))\n",
    "        img = img.convert('RGB')\n",
    "        if self.transform is not None:\n",
    "            img = self.transform(img)\n",
    "        return img\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.img_filename)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = DatasetProcessing('data', 'images')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f1b2882a160>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f1b288be320>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.imshow(test)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python ml",
   "language": "python",
   "name": "ml"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
