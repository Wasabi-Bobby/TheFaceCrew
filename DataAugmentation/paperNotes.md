## Learning Data Augmentation Strategies for Object Detection
- elastic distortions?
   - effect scale, translation, and rotation
   - references in 'Learning Data Augmentation Strategies for Object Detection'
- random cropping
- object-centric cropping
- cut-and-paste images onto other images
- add noise to patches of images
- image mirroring
- multi-scale training (resolution vs scale)
  - artificially pixelate images?
- merging images from same class
  - doesn't sound like it would work
- AutoAugment
  - used by some papers referenced
  
* Seems to be lots of good references
* If we provide bounding box ground truth, we'll have to manipulate those boxes along with the augmentations

## The Effectiveness of Data Augmentation in Image Classification using Deep Learning
- This medium article is an overview of the paper:
  - https://medium.com/@dc.aihub/data-augmentation-for-neural-networks-and-computer-vision-b4f993c34e91
- Look into use of GANs?
- RGB value normalization
- This paper has a few different examples of explicit network definitions
