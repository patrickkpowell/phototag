#!/Users/ppowell/Documents/huggyface/.env/bin/python

import sys
import exiftool

print("Importing huggyface pipeline")
from transformers import pipeline

# Get any existing Subject EXIF and put into file_data
def get_existing_subject(image_path):
  fd = []
  with exiftool.ExifToolHelper() as et:
    metadata = et.get_metadata(image_path)
    for d in metadata:
      try:
        #print('d["XMP:Subject"] is: ', type(d["XMP:Subject"]))
        if not type(d["XMP:Subject"]) is list:
          #print('d["XMP:Subject"] is a string.  Converting to list.')
          d["XMP:Subject"] = d["XMP:Subject"].split()
        for s in d["XMP:Subject"]:
          #print('s is: ', type(s))
          fd.append(s)
      except KeyError:
        print("No existing XMP:Subject tag in file: Adding XMP:Subject from AI predictions")
  return fd

# Get AI predictions
def get_ai_predictions(image_path):
  vc = pipeline(model="google/vit-base-patch16-224")
  preds = vc(images=image_path)
  # Put AI predictions in pred_data list
  pd = []
  #image_path = "with-metadata/{}".format(img)
  for p in preds:
    for e in p["label"].split(","):
      pd.append(e)
  return pd

# Write XMP:Subject to file
def write_xmp_subject(file_path, subject_data):
    # Ensure subject_data is a string
    if not isinstance(subject_data, str):
        subject_data = str(subject_data)

    with exiftool.ExifTool() as et:
        # Set the XMP:Subject tag
        et.execute(f'-XMP:Subject={subject_data}', file_path)



img = sys.argv[1]
# Get existing Subject EXIF and put into file_data
file_data = get_existing_subject(img)
# Get AI predictions and put into pred_data
pred_data = get_ai_predictions(img)


# Combine AI and existing tags
subject = file_data + pred_data
# Remove leading and trailing whitespace within elements of the list
subject = [x.strip() for x in subject]
print('Subject: ', subject)

# Write XMP:Subject to file
write_xmp_subject(img, subject)