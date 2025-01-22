
model_path = 'models/refcoco.pth'
bert_config_path = 'configs/config_bert.json'
use_cuda = False

import sys
from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
# from models.tokenization_bert import BertTokenizer

import torch
from torch import nn
from torchvision import transforms

import json

class VL_Transformer_ITM(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config_bert = ''
                 ):
        super().__init__()
    
        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)   
        
        self.itm_head = nn.Linear(768, 2) 

        
    def forward(self, image, text):
        image_embeds = self.visual_encoder(image) 

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids, 
                                attention_mask = text.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,      
                                return_dict = True,
                               )     
           
        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = self.itm_head(vl_embeddings)   
        return vl_output

import re

def pre_caption(caption,max_words=30):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])            
    return caption

from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt
import torchvision.transforms

def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0: 
        attMap /= attMap.max()
    attMap = torchvision.transforms.functional.resize(attMap.unsqueeze(0), (img.shape[:2]), interpolation=Image.BICUBIC)[0].detach().cpu().numpy()
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform = transforms.Compose([
    transforms.Resize((384,384),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])     

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

albef = VL_Transformer_ITM(text_encoder='bert-base-uncased', config_bert=bert_config_path)

checkpoint = torch.load(model_path, map_location='cpu')              
msg = albef.load_state_dict(checkpoint,strict=False)
albef.eval()

block_num = 8

albef.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

albef.cuda() 

from PIL import Image
import requests
from transformers import SamModel, SamProcessor

sam = SamModel.from_pretrained("facebook/sam-vit-base").cuda()
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


import numpy as np

class BinaryConfusionMatrix:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, y_pred: np.ndarray, y_true: np.ndarray):
        assert y_pred.shape == y_true.shape, "y_pred and y_true must have the same shape " + str(y_pred.shape) + " != " + str(y_true.shape)
        self.tp += np.sum((y_pred == 1) & (y_true == 1))
        self.fp += np.sum((y_pred == 1) & (y_true == 0))
        self.fn += np.sum((y_pred == 0) & (y_true == 1))
        self.tn += np.sum((y_pred == 0) & (y_true == 0))

    def get_iou(self):
        denominator = self.tp + self.fp + self.fn
        if denominator == 0:
            return -0.01
        return self.tp / denominator

    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()
        if precision + recall == 0:
            return -0.01
        return 2 * (precision * recall) / (precision + recall)

    def get_precision(self):
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0

    def get_recall(self):
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0




import os
import cv2
import pylab
root_path = "/mnt/fastdata/marshall/gasvid_val"
# idx = 1000
confusion_matrix = BinaryConfusionMatrix()
files = sorted(os.listdir(os.path.join(root_path, "images")))
videos = list(set([f.split("_")[0] for f in files]))
frames = {}
import random
random.seed(19890604)
for video in videos:
    frames[video] = []
    for f in files:
        if f.split("_")[0] == video:
            frames[video].append(f)
            
    # frames[video] = sorted(random.sample(frames[video], 300))

    




import wandb
wandb.init(project="albef")

for video in videos:
    video_confusion_matrix = BinaryConfusionMatrix()
    for file_name in frames[video]:
        gt = cv2.resize(cv2.imread(os.path.join(root_path, "masks", file_name.replace("jpg", "png")), cv2.IMREAD_GRAYSCALE), (384, 384), interpolation = cv2.INTER_CUBIC) > 0
        if gt.sum() == 0:
            print("Skipping")
            continue
        
        from PIL import Image
        text_input = "red steam (not cloud) coming from a pipe with background of sky and chemical plant"
        text_input = pre_caption(text_input)
        text_input = tokenizer(text_input, return_tensors="pt").to("cuda:0")
        image = Image.open(os.path.join(root_path, "images", file_name))
        softmask = Image.open(os.path.join(root_path, "softmask", file_name))
        # make image where softmask area is more red
        image = np.array(image)
        softmask = np.array(softmask)
        red_image = image.copy().astype(int)
        red_image[:,:,0] = 1.5 * red_image[:,:,0]
        red_image = np.clip(red_image, 0, 255)
        red_image = np.where(softmask > 10, red_image, image)
        red_image = Image.fromarray(red_image.astype(np.uint8))

        image = transform(red_image).unsqueeze(0).to("cuda:0")
        output = albef(image, text_input)
        loss = output[:,1].sum()

        albef.zero_grad()
        loss.backward()    

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

            grads=albef.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
            cams=albef.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask

            gradcam = cams * grads
            gradcam = gradcam[0].mean(0).cpu().detach()
            
        # show the attention map
        num_image = len(text_input.input_ids[0]) + 1 
        # fig, ax = plt.subplots(num_image//3 + 1, 3, figsize=(15,5*num_image))

        # pylab.subplot(num_image//3 + 1 , 3, 1)
        # pylab.imshow(red_image)
        # pylab.xlabel("Image")

        for i,token_id in enumerate(text_input.input_ids[0]):
            word = tokenizer.decode([token_id])
            gradcam_image = getAttMap(np.array(red_image)/255, gradcam[i])
        #     pylab.subplot(num_image//3 + 1 , 3, 1 + i)
        #     pylab.imshow(gradcam_image)
        #     pylab.xlabel(word)
        # pylab.show()
        # 5/0 why show attention map then has good iou
            
        words = [tokenizer.decode([token_id]) for token_id in text_input.input_ids[0]]
        map_of_interest = gradcam[words.index("steam")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.5
        from scipy import ndimage
        from skimage import measure
        labels, n = ndimage.label(segmented)
        com = ndimage.center_of_mass(segmented, labels, range(1, n+1))

        pylab.scatter([c[1] for c in com], [c[0] for c in com], c='b', s=10)
        positive_points =  [[(c[1], c[0]) for c in com]] 

        map_of_interest = gradcam[words.index("red")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.5
        from scipy import ndimage
        from skimage import measure
        labels, n = ndimage.label(segmented)
        com = ndimage.measurements.center_of_mass(segmented, labels, range(1, n+1))

        pylab.scatter([c[1] for c in com], [c[0] for c in com], c='b', s=10)
        positive_points[0].extend([(c[1], c[0]) for c in com])


        x_coords = [point[0] for point in positive_points[0]]
        y_coords = [point[1] for point in positive_points[0]]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        expanded_min_x = min_x - 20
        expanded_max_x = max_x + 20
        expanded_min_y = min_y - 20
        expanded_max_y = max_y + 20

        bounding_box = (expanded_min_x, expanded_min_y, expanded_max_x, expanded_max_y)


        map_of_interest = gradcam[words.index("sky")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.5
        # find the center of mass for each segment
        from scipy import ndimage
        from skimage import measure
        # label each segment
        labels, n = ndimage.label(segmented)
        # get the center of mass
        com = ndimage.measurements.center_of_mass(segmented, labels, range(1, n+1))

        pylab.scatter([c[1] for c in com], [c[0] for c in com], c='b', s=10)
        negative_points =  [[(c[1], c[0]) for c in com]]

        map_of_interest = gradcam[words.index("background")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.5
        # find the center of mass for each segment
        from scipy import ndimage
        from skimage import measure
        # label each segment
        labels, n = ndimage.label(segmented)
        # get the center of mass
        com = ndimage.measurements.center_of_mass(segmented, labels, range(1, n+1))

        negative_points[0].extend([(c[1], c[0]) for c in com])


        all_points = torch.cat([torch.tensor(positive_points), torch.tensor(negative_points)], dim=1)
        labels = torch.tensor([1] * len(positive_points[0]) + [-1] * len(negative_points[0])).unsqueeze(0).unsqueeze(0)

        print(all_points.shape, labels.shape)
        if len(labels[0][0]) == 0:
            pred = np.zeros((384, 384))
        else:
            raw_image = (red_image).resize((384,384)).convert("RGB")
            # raw_image = Image.fromarray(softmask).resize((384,384)).convert("RGB")

            input_points = all_points

            inputs = processor(raw_image, input_points=input_points, input_labels=labels, input_boxes=torch.tensor(bounding_box).unsqueeze(0).unsqueeze(0), return_tensors="pt").to("cuda")
            outputs = sam(**inputs)

            masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            # scores = outputs.iou_scores
            areas = [mask.sum() for mask in masks[0][0]]
            
            best_idx = np.array(areas).argmin()
            
            pred = masks[0][0][best_idx].numpy()
        gt = cv2.resize(cv2.imread(os.path.join(root_path, "masks", file_name.replace("jpg", "png")), cv2.IMREAD_GRAYSCALE), (384, 384), interpolation = cv2.INTER_CUBIC) > 0
        confusion_matrix.update(pred, gt)
        video_confusion_matrix.update(pred, gt)
        pylab.clf()
        pylab.subplot(2, 2, 1)
        pylab.imshow(pred)
        pylab.title("Prediction")
        pylab.axis("off")
        pylab.subplot(2, 2, 2)
        pylab.imshow(gt)
        pylab.title("Ground Truth")
        pylab.axis("off")
        pylab.subplot(2, 2, 3)
        pylab.imshow(cv2.resize(cv2.imread(os.path.join(root_path, "images", file_name)), (384, 384)))
        pylab.scatter([c[0] for c in positive_points[0]], [c[1] for c in positive_points[0]], c='b', s=10)
        pylab.scatter([c[0] for c in negative_points[0]], [c[1] for c in negative_points[0]], c='r', s=10)
        pylab.title("Image")
        pylab.axis("off")
        pylab.subplot(2, 2, 4)
        pylab.imshow(gradcam[words.index("steam")].detach().cpu().numpy())
        pylab.title("Attention")
        pylab.axis("off")
        
        # log the images
        print(f"IOU: {confusion_matrix.get_iou()}, F1: {confusion_matrix.get_f1()}, Precision: {confusion_matrix.get_precision()}, Recall: {confusion_matrix.get_recall()}")
        wandb.log({"IOU": confusion_matrix.get_iou(), "F1": confusion_matrix.get_f1(), "Precision": confusion_matrix.get_precision(), "Recall": confusion_matrix.get_recall(), "image": wandb.Image(pylab.gcf())})
    wandb.log({"video_iou": video_confusion_matrix.get_iou(), "video_f1": video_confusion_matrix.get_f1(), "video_precision": video_confusion_matrix.get_precision(), "video_recall": video_confusion_matrix.get_recall(), "video_name": video})
    print(f"\033[91mVideo IOU: {video_confusion_matrix.get_iou()}, F1: {video_confusion_matrix.get_f1()}, Precision: {video_confusion_matrix.get_precision()}, Recall: {video_confusion_matrix.get_recall()}\033[0m")

