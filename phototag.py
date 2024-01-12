#!/Users/ppowell/Documents/huggyface/.env/bin/python

import sys
import os
import exiftool
import torch
import shutil
from PIL import Image

print("Importing huggyface pipeline")
from transformers import pipeline, DetrImageProcessor, DetrForObjectDetection

# Get any existing Subject EXIF and put into file_data
def get_existing_metadata(image_path):
  fd = []
  dt = ""
  with exiftool.ExifToolHelper() as et:
    metadata = et.get_metadata(image_path)
    for d in metadata:
      try:
        if not type(d["XMP:Subject"]) is list:
          d["XMP:Subject"] = d["XMP:Subject"].split(",")
        for s in d["XMP:Subject"]:
          s = s.replace("'", "")
          s = s.replace("[", "")
          s = s.replace("]", "")
          fd.append(s)
      except KeyError:
        print("No existing XMP:Subject tag in file: Adding XMP:Subject from AI predictions")
      try:
        dt = d["EXIF:DateTimeOriginal"].split(" ")[0].replace(":", "-")
        # Remove leading and trailing whitespace in date_taken
        dt = dt.strip()
      except KeyError:
        print("No EXIF:DateTimeOriginal tag in file")
        dt = "no_date"
      print("Date/Time Original: ", dt)
  # Sort and Unique file_data
  fd = sorted(set(fd))
  # Remove duplicates
  fd = list(dict.fromkeys(fd))
  # Remove empty elements
  fd = list(filter(None, fd))
  # Remove elements with only whitespace
  fd = list(filter(str.strip, fd))
  # Remove leading and trailing whitespace within elements of the list
  fd = [x.strip() for x in fd]
  # Remove ' characters
  fd = [x.replace("'", "") for x in fd]
  # print("Existing XMP:Subject tags: ", fd)
  return fd, dt

# Get AI predictions
def get_ai_predictions(image_path):
  models = []
  print("Gooogle AI")
  google_vc = pipeline(model="google/vit-base-patch16-224")#, task="zero-shot-object-detection")
  google_preds = google_vc(images=image_path)#, candidate_labels=['human face','car','gown','Toyota Land Cruiser'])
  models.append(google_preds)

  # print("Facebook AI")
  # processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
  # model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
  # image = Image.open(img)
  # inputs = processor(images=image, return_tensors="pt")
  # outputs = model(**inputs)
  # # convert outputs (bounding boxes and class logits) to COCO API
  # # let's only keep detections with score > 0.9
  # target_sizes = torch.tensor([image.size[::-1]])
  # results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
 
  # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
  #     box = [round(i, 2) for i in box.tolist()]
  #     print(
  #             f"Detected {model.config.id2label[label.item()]} with confidence "
  #             f"{round(score.item(), 3)} at location {box}"
  #     )

  # Put AI predictions in pred_data list
  pd = []
  for model in models:
    for p in model:
      for e in p["label"].split(","):
        pd.append(e)
    # Sort and Unique pred_data
    pd = sorted(set(pd))
    # Remove duplicates
    pd = list(dict.fromkeys(pd))
    # Remove empty elements
    pd = list(filter(None, pd))
    # Remove elements with only whitespace
    pd = list(filter(str.strip, pd))
    # Remove leading and trailing whitespace within elements of the list
    pd = [x.strip() for x in pd]
    # Remove ' characters
    pd = [x.replace("'", "") for x in pd]
  return pd

# Write XMP:Subject to file
def write_xmp_subject(file_path, subject_data):
    # Ensure subject_data is a string
    if not isinstance(subject_data, str):
        subject_data = str(subject_data)

    # Set the XMP:Subject tag
    with exiftool.ExifTool() as et:
        et.execute(f'-XMP:Subject={subject_data}', file_path)

# Build new Subject String
def build_subject(fd, pd):
  # Create subject list from file_data
  s = fd
  # Set write flag
  write_flag = False
  # Loop through pred_data and add to subject list if not already there
  for p in pd:
    if p not in s:
      print("New ai tag found: ", p)
      s.append(p)
      # Set write flag
      write_flag = True
  # Convert subject list to string
  strSub = ', '.join(s)
  # Remove list brackets and ' characters
  strSub = strSub.replace("'", "")
  strSub = strSub.replace("[", "")
  strSub = strSub.replace("]", "")
  return strSub, write_flag

# Write new Subject String to file if write flag is set
def write_xmp_subject(fp, sd):
    # Ensure subject_data is a string
    if not isinstance(sd, str):
        subject_data = str(sd)

    # Set the XMP:Subject tag
    with exiftool.ExifTool() as et:
        et.execute(f'-XMP:Subject={sd}', fp)

def create_directory(dp):
    if not os.path.exists(dp):
        os.makedirs(dp)
        print(f"Directory created: {dp}")
    else:
        print(f"Directory already exists: {dp}")

# Process args
# TODO:
#       Add  handling for args
#       Add handling for directories
#       Add handling for destination file structure (metatags, dates, etc.)
src_dir = sys.argv[1]
#dst_dir = sys.argv[2]
for filename in os.listdir(src_dir):
  if filename.endswith(".jpg") or filename.endswith(".JPG"):
    img = os.path.join(src_dir, filename)
    print("Processing file: ", img)
    # Get existing Subject EXIF and put into file_data
    file_data, date_taken = get_existing_metadata(img)
    
    # Get AI predictions and put into pred_data
    pred_data = get_ai_predictions(img)
    
    # Set write flag
    write_xmp = False
    
    # Build new Subject String
    strSubject, write_xmp = build_subject(file_data, pred_data)
    
    # Write new Subject String to file if write flag is set
    if write_xmp:
      if not len(sys.argv) < 2:
        print("No destination directory specifiedm  Using ", os.getcwd()+"/ai_processed/ as root directory")
        image_dir = os.getcwd() + "/ai_processed/" + date_taken
      print(image_dir)
      create_directory(image_dir)
      image_path = "{}/{}".format(image_dir, filename)
      
      try:
        shutil.copy(img, image_path)
      except Exception as e:
        print(f"Error occurred while copying file: {e}")
      
      print("Writing new tags to file: ", image_path)
      write_xmp_subject(image_path, strSubject)
      try:
         os.remove(image_path + '_original')
      except Exception as e:
        print(f"Error occurred while deleting copy of original file: {e}")
    else:
      print("No new tags to add")
  else:
    continue

# img = sys.argv[1]

