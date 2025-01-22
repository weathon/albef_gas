
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
# root_path = "../frames"
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
    # sample 300 frames
    # frames[video] = random.sample(frames[video], 500)


import wandb
wandb.init(project="albef")

black_list = [2563, 1473] 
for video in videos:
    if int(video) in black_list:
        print("Blacklisted")
        continue
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
        # increase contrast
        image = np.array(image)
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        image = Image.fromarray(image.astype(np.uint8))
        softmask = Image.open(os.path.join(root_path, "softmask", file_name))
        # make image where softmask area is more red
        image = np.array(image)
        softmask = np.array(softmask)
        red_image = image.copy().astype(int)
        red_image[:,:,0] = 1.5 * red_image[:,:,0]
        red_image = np.clip(red_image, 0, 255)
        red_image = np.where(softmask > 3, red_image, image)
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
        segmented = map_of_interest > 0.6
        from scipy import ndimage
        from skimage import measure
        labels, n = ndimage.label(segmented)
        com = ndimage.center_of_mass(segmented, labels, range(1, n+1))

        pylab.scatter([c[1] for c in com], [c[0] for c in com], c='b', s=10)
        positive_points =  [[(c[1], c[0]) for c in com]] 

        map_of_interest = gradcam[words.index("red")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.6
        from scipy import ndimage
        from skimage import measure
        labels, n = ndimage.label(segmented)
        com = ndimage.measurements.center_of_mass(segmented, labels, range(1, n+1))

        pylab.scatter([c[1] for c in com], [c[0] for c in com], c='b', s=10)
        positive_points[0].extend([(c[1], c[0]) for c in com])

        map_of_interest = gradcam[words.index("sky")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.6
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
        segmented = map_of_interest > 0.6
        # find the center of mass for each segment
        from scipy import ndimage
        from skimage import measure
        # label each segment
        labels, n = ndimage.label(segmented)
        # get the center of mass
        com = ndimage.measurements.center_of_mass(segmented, labels, range(1, n+1))

        negative_points[0].extend([(c[1], c[0]) for c in com])

        map_of_interest = gradcam[words.index("pipe")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.6
        # find the center of mass for each segment
        from scipy import ndimage
        from skimage import measure
        # label each segment
        labels, n = ndimage.label(segmented)
        # get the center of mass
        com = ndimage.measurements.center_of_mass(segmented, labels, range(1, n+1))

        negative_points[0].extend([(c[1], c[0]) for c in com])


        # use background frame and positive terms to detect false positive in original image. 
        # Either remove them if too close (like bbox) did not think about this, or added as negative prompt, or both
        background = transform(Image.open(os.path.join(root_path, "backgrounds", file_name)).convert("RGB")).unsqueeze(0).to("cuda:0")
        output = albef(background, text_input)
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
            
        num_image = len(text_input.input_ids[0]) + 1 
       

        for i,token_id in enumerate(text_input.input_ids[0]):
            word = tokenizer.decode([token_id])
            gradcam_image = getAttMap(np.array(red_image)/255, gradcam[i])
            
        words = [tokenizer.decode([token_id]) for token_id in text_input.input_ids[0]]
        map_of_interest = gradcam[words.index("steam")].detach().cpu().numpy()
        map_of_interest = cv2.resize(map_of_interest, (384, 384), interpolation = cv2.INTER_CUBIC)  
        map_of_interest = cv2.blur(map_of_interest, (10,10))
        segmented = map_of_interest > 0.6
        from scipy import ndimage
        from skimage import measure
        labels, n = ndimage.label(segmented)
        com = ndimage.center_of_mass(segmented, labels, range(1, n+1))
        background_positive = [[(c[1], c[0]) for c in com]]
        
        new_positive_points = []
        removed_positive_points = []
        for i in positive_points[0]:
            # if it is too close (10px) to false positive, ignore it, otherwise add it to new positive points
            too_close = False
            for j in background_positive[0]:
                if np.linalg.norm(np.array(i) - np.array(j)) < 20:
                    too_close = True
                    break
            if not too_close:
                new_positive_points.append(i)
            else:
                removed_positive_points.append(i)
        positive_points_center = np.array(positive_points[0]).mean(0)
        positive_points_std = np.array(positive_points[0]).std(0)
        # remove points that are 1.5 std away from the center
        filtered_positive_points = []
        for point in positive_points[0]:
            if (abs(point[0] - positive_points_center[0]) <= 10 * positive_points_std[0]) and (abs(point[1] - positive_points_center[1]) <= 10 * positive_points_std[1]):
                filtered_positive_points.append(point)
            else:
                removed_positive_points.append(point)
        positive_points[0] = filtered_positive_points #yao check continuties
        
        positive_points = [new_positive_points]
        removed_positive_points = [removed_positive_points]
    

        all_points = []
        labels = []
        if len(positive_points[0]) > 0:
            all_points.extend(positive_points[0])
            labels.extend([1] * len(positive_points[0]))
        if len(negative_points[0]) > 0:
            all_points.extend(negative_points[0])
            labels.extend([-1] * len(negative_points[0]))
        if len(background_positive[0]) > 0:
            all_points.extend(background_positive[0])
            labels.extend([0] * len(background_positive[0]))
        if len(removed_positive_points[0]) > 0:
            all_points.extend(removed_positive_points[0])
            labels.extend([0] * len(removed_positive_points[0]))
        # print(all_points, labels)
        all_points = torch.tensor(all_points).unsqueeze(0)
        labels = torch.tensor(labels).unsqueeze(0).unsqueeze(0)

        print(all_points.shape, labels.shape)
        if len(positive_points[0]) == 0: # should check positive points not all points
            pred = np.zeros((384, 384))
        else:
            # only box when it has points, forget this before
            x_coords = [point[0] for point in positive_points[0]]
            y_coords = [point[1] for point in positive_points[0]]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            expanded_min_x = min_x - 20
            expanded_max_x = max_x + 20
            expanded_min_y = min_y - 20
            expanded_max_y = max_y + 20

            bounding_box = (expanded_min_x, expanded_min_y, expanded_max_x, expanded_max_y)
            # raw_image = (red_image).resize((384,384)).convert("RGB")
            raw_image = Image.fromarray(softmask).resize((384,384)).convert("RGB")
            # increase contrast 
            raw_image = np.array(raw_image)
            raw_image = cv2.convertScaleAbs(raw_image, alpha=4, beta=0)
            raw_image = Image.fromarray(raw_image.astype(np.uint8))


            input_points = all_points

            inputs = processor(raw_image, input_points=input_points, input_labels=labels.sign(), input_boxes=torch.tensor(bounding_box).unsqueeze(0).unsqueeze(0), return_tensors="pt").to("cuda")
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
        pylab.imshow(red_image.resize((384,384)))
        # label_names = {1: "Positive", -1: "Negative", -2: "False Positive", 0: "Removed Positive"}
        # # pylab.scatter([c[0] for c in all_points[0]], [c[1] for c in all_points[0]], labels=[label_names[l.item()] for l in labels[0][0]])
        # for label in label_names.keys():
        #     points = [all_points[0][i] for i in range(len(all_points[0])) if labels[0][0][i] == label]
        #     if len(points) > 0:
        #         pylab.scatter([c[0] for c in points], [c[1] for c in points], label=label_names[label])
        pylab.scatter([c[0] for c in all_points[0]], [c[1] for c in all_points[0]], c=[label.item() for label in labels[0][0]])
        # draw bounding_box
        pylab.gca().add_patch(pylab.Rectangle((bounding_box[0], bounding_box[1]), bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1], linewidth=1, edgecolor='r', facecolor='none'))
        pylab.legend()
            
        pylab.title("Image")
        pylab.axis("off")
        pylab.subplot(2, 2, 4)
        # pylab.imshow(gradcam[words.index("steam")].detach().cpu().numpy())
        # pylab.title("Attention")
        pylab.imshow(raw_image)
        pylab.title("Softmask")
        pylab.axis("off")
        
        # log the images
        print(f"IOU: {confusion_matrix.get_iou()}, F1: {confusion_matrix.get_f1()}, Precision: {confusion_matrix.get_precision()}, Recall: {confusion_matrix.get_recall()}")
        wandb.log({"IOU": confusion_matrix.get_iou(), "F1": confusion_matrix.get_f1(), "Precision": confusion_matrix.get_precision(), "Recall": confusion_matrix.get_recall(), "image": wandb.Image(pylab.gcf())})
    wandb.log({"video_iou": video_confusion_matrix.get_iou(), "video_f1": video_confusion_matrix.get_f1(), "video_precision": video_confusion_matrix.get_precision(), "video_recall": video_confusion_matrix.get_recall(), "video_name": video})
    print(f"\033[91mVideo IOU: {video_confusion_matrix.get_iou()}, F1: {video_confusion_matrix.get_f1()}, Precision: {video_confusion_matrix.get_precision()}, Recall: {video_confusion_matrix.get_recall()}\033[0m")

