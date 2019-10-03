"""Dataset setting and data loader for CUB-200."""

import logging
import os
import tarfile

import torch.utils.data as data
import torchvision
from PIL import Image

class_labels = [
    'Black_footed_Albatross',
    'Laysan_Albatross',
    'Sooty_Albatross',
    'Groove_billed_Ani',
    'Crested_Auklet',
    'Least_Auklet',
    'Parakeet_Auklet',
    'Rhinoceros_Auklet',
    'Brewer_Blackbird',
    'Red_winged_Blackbird',
    'Rusty_Blackbird',
    'Yellow_headed_Blackbird',
    'Bobolink',
    'Indigo_Bunting',
    'Lazuli_Bunting',
    'Painted_Bunting',
    'Cardinal',
    'Spotted_Catbird',
    'Gray_Catbird',
    'Yellow_breasted_Chat',
    'Eastern_Towhee',
    'Chuck_will_Widow',
    'Brandt_Cormorant',
    'Red_faced_Cormorant',
    'Pelagic_Cormorant',
    'Bronzed_Cowbird',
    'Shiny_Cowbird',
    'Brown_Creeper',
    'American_Crow',
    'Fish_Crow',
    'Black_billed_Cuckoo',
    'Mangrove_Cuckoo',
    'Yellow_billed_Cuckoo',
    'Gray_crowned_Rosy_Finch',
    'Purple_Finch',
    'Northern_Flicker',
    'Acadian_Flycatcher',
    'Great_Crested_Flycatcher',
    'Least_Flycatcher',
    'Olive_sided_Flycatcher',
    'Scissor_tailed_Flycatcher',
    'Vermilion_Flycatcher',
    'Yellow_bellied_Flycatcher',
    'Frigatebird',
    'Northern_Fulmar',
    'Gadwall',
    'American_Goldfinch',
    'European_Goldfinch',
    'Boat_tailed_Grackle',
    'Eared_Grebe',
    'Horned_Grebe',
    'Pied_billed_Grebe',
    'Western_Grebe',
    'Blue_Grosbeak',
    'Evening_Grosbeak',
    'Pine_Grosbeak',
    'Rose_breasted_Grosbeak',
    'Pigeon_Guillemot',
    'California_Gull',
    'Glaucous_winged_Gull',
    'Heermann_Gull',
    'Herring_Gull',
    'Ivory_Gull',
    'Ring_billed_Gull',
    'Slaty_backed_Gull',
    'Western_Gull',
    'Anna_Hummingbird',
    'Ruby_throated_Hummingbird',
    'Rufous_Hummingbird',
    'Green_Violetear',
    'Long_tailed_Jaeger',
    'Pomarine_Jaeger',
    'Blue_Jay',
    'Florida_Jay',
    'Green_Jay',
    'Dark_eyed_Junco',
    'Tropical_Kingbird',
    'Gray_Kingbird',
    'Belted_Kingfisher',
    'Green_Kingfisher',
    'Pied_Kingfisher',
    'Ringed_Kingfisher',
    'White_breasted_Kingfisher',
    'Red_legged_Kittiwake',
    'Horned_Lark',
    'Pacific_Loon',
    'Mallard',
    'Western_Meadowlark',
    'Hooded_Merganser',
    'Red_breasted_Merganser',
    'Mockingbird',
    'Nighthawk',
    'Clark_Nutcracker',
    'White_breasted_Nuthatch',
    'Baltimore_Oriole',
    'Hooded_Oriole',
    'Orchard_Oriole',
    'Scott_Oriole',
    'Ovenbird',
    'Brown_Pelican',
    'White_Pelican',
    'Western_Wood_Pewee',
    'Sayornis',
    'American_Pipit',
    'Whip_poor_Will',
    'Horned_Puffin',
    'Common_Raven',
    'White_necked_Raven',
    'American_Redstart',
    'Geococcyx',
    'Loggerhead_Shrike',
    'Great_Grey_Shrike',
    'Baird_Sparrow',
    'Black_throated_Sparrow',
    'Brewer_Sparrow',
    'Chipping_Sparrow',
    'Clay_colored_Sparrow',
    'House_Sparrow',
    'Field_Sparrow',
    'Fox_Sparrow',
    'Grasshopper_Sparrow',
    'Harris_Sparrow',
    'Henslow_Sparrow',
    'Le_Conte_Sparrow',
    'Lincoln_Sparrow',
    'Nelson_Sharp_tailed_Sparrow',
    'Savannah_Sparrow',
    'Seaside_Sparrow',
    'Song_Sparrow',
    'Tree_Sparrow',
    'Vesper_Sparrow',
    'White_crowned_Sparrow',
    'White_throated_Sparrow',
    'Cape_Glossy_Starling',
    'Bank_Swallow',
    'Barn_Swallow',
    'Cliff_Swallow',
    'Tree_Swallow',
    'Scarlet_Tanager',
    'Summer_Tanager',
    'Artic_Tern',
    'Black_Tern',
    'Caspian_Tern',
    'Common_Tern',
    'Elegant_Tern',
    'Forsters_Tern',
    'Least_Tern',
    'Green_tailed_Towhee',
    'Brown_Thrasher',
    'Sage_Thrasher',
    'Black_capped_Vireo',
    'Blue_headed_Vireo',
    'Philadelphia_Vireo',
    'Red_eyed_Vireo',
    'Warbling_Vireo',
    'White_eyed_Vireo',
    'Yellow_throated_Vireo',
    'Bay_breasted_Warbler',
    'Black_and_white_Warbler',
    'Black_throated_Blue_Warbler',
    'Blue_winged_Warbler',
    'Canada_Warbler',
    'Cape_May_Warbler',
    'Cerulean_Warbler',
    'Chestnut_sided_Warbler',
    'Golden_winged_Warbler',
    'Hooded_Warbler',
    'Kentucky_Warbler',
    'Magnolia_Warbler',
    'Mourning_Warbler',
    'Myrtle_Warbler',
    'Nashville_Warbler',
    'Orange_crowned_Warbler',
    'Palm_Warbler',
    'Pine_Warbler',
    'Prairie_Warbler',
    'Prothonotary_Warbler',
    'Swainson_Warbler',
    'Tennessee_Warbler',
    'Wilson_Warbler',
    'Worm_eating_Warbler',
    'Yellow_Warbler',
    'Northern_Waterthrush',
    'Louisiana_Waterthrush',
    'Bohemian_Waxwing',
    'Cedar_Waxwing',
    'American_Three_toed_Woodpecker',
    'Pileated_Woodpecker',
    'Red_bellied_Woodpecker',
    'Red_cockaded_Woodpecker',
    'Red_headed_Woodpecker',
    'Downy_Woodpecker',
    'Bewick_Wren',
    'Cactus_Wren',
    'Carolina_Wren',
    'House_Wren',
    'Marsh_Wren',
    'Rock_Wren',
    'Winter_Wren',
    'Common_Yellowthroat',
]


class CUB200(data.Dataset):
    r"""Caltech-UCSD-Birds-200-2011 Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, use the training split.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``

    See http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    """

    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    num_classes = 200
    num_images = 11788

    """
    This contains:
      - attributes.txt
      - CUB_200_2011/
        - attributes/
          - ...
        - images/
          - [CLS] ...
            - [IMG] ...
        - parts/
          - ...
        - bounding_boxes.txt
        - classes.txt
          Each row: [CLS_IDX (1-indexed)] [CLS]
        - images.txt
          Each row: [IMG_IDX (1-indexed)] [CLS]/[IMG]
        - image_class_labels.txt
          Each row: [IMG_IDX (1-indexed)] [CLS_IDX (1-indexed)]
        - train_test_split.txt
          Each row: [IMG_IDX (1-indexed)] [IS_TRAINING_IMG]
        - README
    """

    def __init__(self, root, train=True, transform=None, download=False):
        # init params
        self.root = os.path.expanduser(root)
        self.filename = self.url.rpartition('/')[2]
        self.extract_dirname = "CUB_200_2011"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            try:
                self.download_and_extract()
            except FileExistsError:
                pass
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.load()

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return len(self.images)

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.extract_dirname))

    def download_and_extract(self):
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        if not self._check_exists():
            torchvision.datasets.utils.download_url(self.url, root=self.root, filename=self.filename, md5=None)
            fullpath = os.path.abspath(os.path.join(self.root, self.filename))
            logging.info("Extract {}".format(fullpath))
            with tarfile.open(fullpath, 'r:gz') as tar:
                tar.extractall(self.root)
            logging.info("[DONE]")

    def load(self):
        """Load dataset."""
        # parse
        def parse(filename, expect_N=None):
            fullpath = os.path.join(self.root, self.extract_dirname, filename)
            res = []
            with open(fullpath, 'r') as f:
                for l in f.readlines():
                    split = l.strip().split()
                    if len(split) > 0:
                        assert len(split) == 2
                        assert int(split[0]) == len(res) + 1  # 1-indexed
                        res.append(split[1])
            if expect_N is not None:
                assert len(res) == expect_N
            return tuple(res)

        self.class_names = parse('classes.txt', self.num_classes)
        image_paths = parse('images.txt', self.num_images)
        image_class_labels = parse('image_class_labels.txt', self.num_images)
        train_test_split = parse('train_test_split.txt', self.num_images)

        filtered_image_paths = []
        filtered_labels = []

        for image_path, label, is_train in zip(image_paths, image_class_labels, train_test_split):
            if bool(int(is_train)) == self.train:
                filtered_image_paths.append(os.path.join(self.root, self.extract_dirname, 'images', image_path))
                filtered_labels.append(int(label) - 1)  # to 0-indexed
        self.images = tuple(Image.open(p).convert('RGB') for p in filtered_image_paths)
        self.labels = tuple(filtered_labels)
