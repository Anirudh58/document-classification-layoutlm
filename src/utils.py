import os
import sys

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import pytesseract

import torch
from transformers import LayoutLMTokenizer

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

def normalize_box(box, width, height):
     """
        Given a box and image dimensions, normalize the box.
        Follows the instructions from the LayoutLM paper.
     """
     return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]

def draw_boxes(image, boxes, color='red'):
    """
        Given an image and boxes, draw the boxes on the image.
        Args:
            image (PIL.Image): Image.
            boxes (list): List of actual boxes.
            color (string): Color of the boxes.
    """

    image = image.convert('RGB')
    draw = ImageDraw.Draw(image, mode='RGB')
    for box in boxes:
        draw.rectangle(box, outline=color)
    return image

def apply_ocr(image_path):
    """
        Given an image path, apply OCR and return the words and boxes.
        Args:
            image_path (string): Path to the image.
        Returns:
            output (dict): Dictionary containing the words and boxes.
                - words (list): List of words.
                - nboxes (list): List of normalized boxes.
                - aboxes (list): List of actual boxes.
    """

    image = Image.open(image_path)

    # get shape
    width, height = image.size
    
    # apply ocr and clean
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    words = list(ocr_df.text)
    words = [str(w) for w in words]
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row) 
        actual_box = [x, y, x+w, y+h] 
        actual_boxes.append(actual_box)

    # normalize boxes
    normalized_boxes = [normalize_box(box, width, height) for box in actual_boxes]

    # output
    output = {
        'words': words,
        'nboxes': normalized_boxes,
        'aboxes': actual_boxes,
    }

    return output

def encode_document(document, max_seq_length=512, pad_token_box=[0, 0, 0, 0]):
    """
        Given a document, encode it.
        Args:
            document (dict): Dictionary containing the words and boxes.
                - words (list): List of words.
                - nboxes (list): List of normalized boxes.
                - aboxes (list): List of actual boxes.
                - label (int): Label.
            max_seq_length (int): Maximum sequence length.
            pad_token_box (list): Padding token box.
    """

    # unpack document
    words, nboxes, label = document['words'], document['nboxes'], document['label']

    # check if number of words and boxes are equal
    assert len(words) == len(nboxes)

    # tokenize words
    token_boxes = []
    for word, box in zip(words, nboxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))

    # Account for [CLS] and [SEP] with "- 2"
    special_tokens_count = 2
    if len(token_boxes) > max_seq_length - special_tokens_count:
        token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    # encode
    encoding = tokenizer(
        ' '.join(words),
        padding='max_length',
        truncation=True,
    )
    input_ids = tokenizer(' '.join(words), truncation=True)['input_ids']
    padding_length = max_seq_length - len(input_ids)
    token_boxes = token_boxes + ([pad_token_box] * padding_length)
    encoding['bbox'] = token_boxes
    encoding['label'] = label

    # check if shapes of masks and bounding boxes are correct
    assert len(encoding['input_ids']) == max_seq_length
    assert len(encoding['attention_mask']) == max_seq_length
    assert len(encoding['token_type_ids']) == max_seq_length
    assert len(encoding['bbox']) == max_seq_length

    return encoding
