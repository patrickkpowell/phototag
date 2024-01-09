#!/Users/ppowell/Documents/huggyface/.env/bin/python

import sys
import exiftool

print("Importing huggyface pipeline")
from transformers import pipeline

print("Importing exif and pillow")
from exif import Image as ExifImage
from PIL import Image as PillowImage
from PIL import ExifTags

img = sys.argv[1]

# Get AI predictions
vc = pipeline(model="google/vit-base-patch16-224")
preds = vc(images=img)

# Put AI predictions in pred_data list
pred_data = []
#image_path = "with-metadata/{}".format(img)
for p in preds:
  for e in p["label"].split(","):
    pred_data.append(e)

# Get any existing Subject EXIF and put into file_data
file_data = []
with exiftool.ExifToolHelper() as et:
  metadata = et.get_metadata(img)
  for d in metadata:
    try:
      for s in d["XMP:Subject"]:
        file_data.append(s)
    except KeyError:
      print("No Subject Tag in file: Adding XMP:Subject")
      # Add Tag here

# Combine AI and existing tags
subject = pred_data + file_data

print('Subject: ', subject)
