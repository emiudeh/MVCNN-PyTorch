from torch.utils.data.dataset import Dataset
import os
from PIL import Image

class MultiViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform

        selection = [".2.", ".5.", ".8.", ".11."]
        selection = ["0", "1", "2", "3"]
        # root / <label>  / <train/test> / <item> / <view>.png
        for label in os.listdir(root): # Label
            views = []
            for view in os.listdir(root + '/' + label):
                # to use specific orientations, comment out the views appending code below
                # and uncomment the if-statement that follows. 
                # selected orientation is specified by the selection variable above.
                # views.append(root + '/' + label + '/' + data_type + '/' + item + '/' + view)

                if any(sel_view in view for sel_view in selection):
                    views.append(root + '/' + label + '/' + view)
                    print(view)

            self.x.append(views)
            self.y.append(self.class_to_idx[label])

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for view in orginal_views:
            im = Image.open(view)
            im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)