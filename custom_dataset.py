from paddle.vision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
import numpy as np
import paddle
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

class CustomDataset(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, target_num=None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 ):
        super(CustomDataset, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform)
        classes, class_to_idx = self.find_classes(self.root)
        self.samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS, is_valid_file)
        self.imgs = self.samples
        self.target_num = target_num
        # breakpoint()
        self.target_transform = target_transform
        self.targets = [s[1] for s in self.samples]

    def make_dataset(self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)

    def __getitem__(self, index):
        path, target = self.samples[index]
        # breakpoint()
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if self.target_num is None:
            while True:
                path2, target2 = self.samples[np.random.choice(len(self.samples), 1)[0]]
                if target == target2:
                    pass
                else:
                    break
            sample2 = self.loader(path2)
            if self.transform is not None:
                sample2 = self.transform(sample2)
            if self.target_transform is not None:
                target2 = self.target_transform(target2)

            sample = paddle.concat((sample, sample2), 0)
            target = paddle.to_tensor([target, target2])

        else:
            samples_array = np.array(self.samples)
            target_list = samples_array[samples_array[:, 1].astype(np.int) == target]
            idx = np.random.choice(target_list.shape[0], self.target_num - 1)
            path2, target2 = target_list[idx, 0], target_list[idx, 1]
            target = paddle.to_tensor([target])
            for p, t in zip(path2, target2):
                sample2 = self.loader(p)
                if self.transform is not None:
                    sample2 = self.transform(sample2)
                if self.target_transform is not None:
                    target2 = self.target_transform(target2)

                sample = paddle.concat((sample, sample2), 0)
                target = paddle.concat((target, paddle.to_tensor(target2.astype(np.long))), 0)

        return sample, target



class CustomDataset_v1(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, target_num=None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 ):
        super(CustomDataset_v1, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform)
        classes, class_to_idx = self.find_classes(self.root)
        self.samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS, is_valid_file)
        self.imgs = self.samples
        self.target_num = target_num
        # breakpoint()
        self.target_transform = target_transform
        self.targets = [s[1] for s in self.samples]

    def make_dataset(self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target