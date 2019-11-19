# TheFaceCrew
Facial recognition with PyTorch!

## Newton Access Notes
- username is cap5415.student##
- I needed to connect via VPN from home
- I had to change the permissions of the `id_rsa_1` file using `sudo chmod 600 cap5415.student36_id_rsa_1` 
- Newton had an anaconda3 module with pytorch already installed
- We may be sharing compute resources with the entire class or part of it (and whoever student29 is has already used ~25% of resources)

## Facial Feature Detection Notes
Dlib, opencv and python should be used for this
Use of dlib to draw the bounding box of the face and use of opencv to obtain the important facial features that stick out
https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ Great article on how to use these libraries to extract facial information as well as a bit of data augmentation
Use of cv is important to actually extract facial features of the picture. Can be used to extract key points of the face (Need a method to classify some of this information however)
Using these points we could perhaps also draw ideas about what points are the most important as well (such as the middle point of the bounding box around the face should always be the nose) We know the points closest to the edge of the box relate to face, the points closest to the top should be the eyes, etc...
Perhaps we should make it mainly recognize the front facing image and not focus too much on the sides of a face. Maybe make multiple CNN's and combine their statistical information to form a solid truth about whether or not a picture belongs to a certain person

https://www.codesofinterest.com/2017/04/extracting-individual-facial-features-dlib.html Another article detailing on how to get certain facial features from people. Actually describes the 68 points given and what each one means. Now with this we can really work on how to extract some key features. Maybe look up some slider functions for character customizations to get an idea on what facial features are really important.

Another idea we can do is to extract the facial type (round, rectangle, etc...). Not very important to implement given the small dataset we have, but it could work.

## Design Ideas
We should each put our respective methods (data augmentation, feature extraction) into seperate python scripts. Once we move onto developing CNNs, we can each create a 'train.py' that imports those scripts and uses them however we want. We can customize our own networks and experiment with different features/augmentations without having to worry about file conflicts in git.

For the data augmentation script, I might have one callable function that takes a data set and a series of bools as parameters. Setting a specific parameter to true means that you want to add that augmentation to the data set. Default values of bools are false, so calling the function with only the data set will return the data set unaltered.
