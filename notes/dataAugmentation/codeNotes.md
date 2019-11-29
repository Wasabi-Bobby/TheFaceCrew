# Code Notes

### Dataset class
In HW4 I made a class for handling the CIFAR dataset so that it would play nicely
with pytorch classes. We can do the same for our dataset.

The class needs to define `__len__(self)` and `__getitem__(self, idx)`

If we keep all of the images in a list or array, then we can pass in the array to the class
on initialization.
    Looks like the pickled CIFAR data came as a dict.

Maybe we can look at pickling datasets for our save configurations.

----

We can keep the dataset as a dict with properties:
    **data**
    **lables**
We don't need a property for label_names because our labels are parts of the filenames.


