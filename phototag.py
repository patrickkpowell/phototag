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
          print('s value: ', s)
          fd.append(s)
      except KeyError:
        print("No existing XMP:Subject tag in file: Adding XMP:Subject from AI predictions")
  print("Existing XMP:Subject tags: ", fd)
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
# Sort and Unique file_data
file_data = sorted(set(file_data))
# Remove duplicates
file_data = list(dict.fromkeys(file_data))
# Remove empty elements
file_data = list(filter(None, file_data))
# Remove elements with only whitespace
file_data = list(filter(str.strip, file_data))
# Remove leading and trailing whitespace within elements of the list
file_data = [x.strip() for x in file_data]
# Remove ' characters
file_data = [x.replace("'", "") for x in file_data]

# Get AI predictions and put into pred_data
pred_data = get_ai_predictions(img)
# Sort and Unique pred_data
pred_data = sorted(set(pred_data))
# Remove duplicates
pred_data = list(dict.fromkeys(pred_data))
# Remove empty elements
pred_data = list(filter(None, pred_data))
# Remove elements with only whitespace
pred_data = list(filter(str.strip, pred_data))
# Remove leading and trailing whitespace within elements of the list
pred_data = [x.strip() for x in pred_data]

subject = file_data
write_xmp = False
for p in pred_data:
  if p not in subject:
    print("New ai tag found: ", p)
    subject.append(p)
    write_xmp = True
strSubject = ', '.join(subject)
print('Initial Subject String: ', strSubject)
# Remove list brackets and ' characters
strSubject = strSubject.replace("'", "")
strSubject = strSubject.replace("[", "")
strSubject = strSubject.replace("]", "")
print('Processed Subject String: ', strSubject)
if write_xmp:
  # Write XMP:Subject to file
  write_xmp_subject(img, subject)
else:
  print("No new tags to add")