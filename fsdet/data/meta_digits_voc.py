import numpy as np

import xml.etree.ElementTree as ET
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pathlib import Path

__all__ = ["register_meta_digits_voc"]


def load_filtered_voc_instances(
    name: str, dirname: Path, split: str, classnames: str
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    is_shots = "shot" in name
    dirname = Path(dirname)
    if is_shots:
        fileids = {}
        split_dir = Path("datasets").joinpath("digit_vocsplit")
        if "seed" in name:
            shot = name.split("_")[-2].split("shot")[0]  # digits_voc_1shot_seed1
            seed = int(name.split("_seed")[-1])
            split_dir = split_dir.joinpath("seed{}".format(seed))
        else:
            shot = name.split("_")[-1].split("shot")[0]
        for cls in classnames:
            with open(split_dir.joinpath("box_{}shot_{}_train.txt".format(shot, cls)), 'rt') as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [Path(fid).stem for fid in fileids_]
                print(fileids_)
                fileids[cls] = fileids_
    else:
        with dirname.joinpath("ImageSets", "Main", split + ".txt") as f:
            fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    if is_shots:
        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                dirname = Path("datasets").joinpath("voc_digits")
                anno_file = dirname.joinpath("Annotations", fileid + ".xml")
                jpeg_file = dirname.joinpath("JPEGImages", fileid + ".png")

                tree = ET.parse(anno_file)

                for obj in tree.findall("object"):
                    r = {
                        "file_name": str(jpeg_file),
                        "image_id": fileid,
                        "height": int(tree.findall("./size/height")[0].text),
                        "width": int(tree.findall("./size/width")[0].text),
                    }
                    cls_ = obj.find("name").text
                    if cls != cls_:
                        continue
                    bbox = obj.find("bndbox")
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ["xmin", "ymin", "xmax", "ymax"]
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [
                        {
                            "category_id": classnames.index(cls),
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                        }
                    ]
                    r["annotations"] = instances
                    dicts_.append(r)
            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        for fileid in fileids:
            anno_file = dirname.joinpath("Annotations", fileid + ".xml")
            jpeg_file = dirname.joinpath("JPEGImages", fileid + ".png")

            tree = ET.parse(anno_file)

            r = {
                "file_name": str(jpeg_file),
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                bbox = obj.find("bndbox")
                bbox = [
                    float(bbox.find(x).text)
                    for x in ["xmin", "ymin", "xmax", "ymax"]
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append(
                    {
                        "category_id": classnames.index(cls),
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                    }
                )
            r["annotations"] = instances
            dicts.append(r)
    return dicts


def register_meta_digits_voc(name, metadata, dirname, split):
    DatasetCatalog.register(
        name,
        lambda: load_filtered_voc_instances(name, dirname, split, metadata["classes"]),
    )

    MetadataCatalog.get(name).set(
        dirname=dirname,
        split=split,
        classes=metadata["classes"],
    )
