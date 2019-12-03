``````````````````
PYTHON VERSION
``````````````````

We used python 3.6.x or 3.7.x
Make sure to have a 64-bit version or pytorch will literally not run at all

``````````````````
PYTORCH VERSION
``````````````````

Most recent version from the website

````````````````````
Libraries required
````````````````````

Numpy - latest version or scikit-image won't work
scikit-image
Pillow (or PIL)
os
matplotlib
scipy
cv2
collections
glob
skimage
cmake (required for dlib and MUST be installed first)
dlib

``````````````````
What needs to be grabbed from our github link or found online to put into data\dlib-models

shape_predictor_68_face_landmarks.dat

``````````````````

If our github link works then go to data then dlib models and grab the shape_predictor_68_face_landmarks.dat file

Otherwise grab from this link

https://github.com/davisking/dlib-models

Download this zip and extract from shape_predictor_68_face_landmarks.dat.bz2
Throw the shape_predictor_68_face_landmarks.dat into data->dlib-models


``````````````````

Final step

``````````````````

Open up your favorite python IDE and run the file named "Execute_This.py" using python 3.x.x and thats it to demo our project


