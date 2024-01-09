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
        if not type(d["XMP:Subject"]) is list:
          d["XMP:Subject"] = d["XMP:Subject"].split(",")
        for s in d["XMP:Subject"]:
          s = s.replace("'", "")
          s = s.replace("[", "")
          s = s.replace("]", "")
          # print('s value: ', s)
          fd.append(s)
      except KeyError:
        print("No existing XMP:Subject tag in file: Adding XMP:Subject from AI predictions")
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
  return pd

# Write XMP:Subject to file
def write_xmp_subject(file_path, subject_data):
    # Ensure subject_data is a string
    if not isinstance(subject_data, str):
        subject_data = str(subject_data)

    # Set the XMP:Subject tag
    with exiftool.ExifTool() as et:
        et.execute(f'-XMP:Subject={subject_data}', file_path)


# Process args
# TODO: Add  handling for args
img = sys.argv[1]

# Get existing Subject EXIF and put into file_data
file_data = get_existing_subject(img)

# Get AI predictions and put into pred_data
pred_data = get_ai_predictions(img)

# Build new Subject String
# Create subject list from file_data
subject = file_data
# Set write flag
write_xmp = False
# Loop through pred_data and add to subject list if not already there
for p in pred_data:
  if p not in subject:
    print("New ai tag found: ", p)
    subject.append(p)
    # Set write flag
    write_xmp = True
# Convert subject list to string
strSubject = ', '.join(subject)
# Remove list brackets and ' characters
strSubject = strSubject.replace("'", "")
strSubject = strSubject.replace("[", "")
strSubject = strSubject.replace("]", "")
# Write new Subject String to file if write flag is set
if write_xmp:
  write_xmp_subject(img, subject)
  print("New tags written")
else:
  print("No new tags to add")