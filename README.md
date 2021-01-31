# show-attend-and-tell-pytorch
Show, attend and tell implementation on pytorch(only supports soft attention)
[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
Dataset used in this pytorch implementation is [Flickr8k dataset]()
## References
- Image Captioning with Flickr8k <https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch>

## Dataset
- Images : <https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip>
- Caption : <https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip>

## How to train/infer
> python main.py --root_dir="root directory" --data_path="path where flickr8k is downloaded" --caption_path="path where caption exists"

## Results
- BLEU-1: 0.633990
- BLEU-2: 0.578760
- BLEU-3: 0.401943
- BLEU-4: 0.161558
