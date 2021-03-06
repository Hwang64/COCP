# COCP

Well-annotated training samples show necessity in achieving high performance of object detection, but collection of massive samples is extremely laborious and costly. Recently, cut-paste based methods show the potential to augment the training samples by cutting the foreground instances and pasting them on some background regions. However, existing cut-paste based methods hardly guarantee the quality of synthetic images due to lack of mechanism to ensure rationality of the pasted instances (e.g., context, geometry and  diversity), limiting the effectiveness of data augmentation. To overcome above issues, this paper proposes a novel Constrained Online Cut-Paste (COCP) method, making an attempt to effectively and efficiently augment training data for improving performance of object detection. Specifically, our COCP generates synthetic images by switching instances of same class from various image pairs in each training mini-batch, ensuring context coherence between the cut instances and the pasted backgrounds. Furthermore, two constraints based on geometric consistency and sample diversity are developed to eliminate counterproductive and meaningless switched instances those suffer from significant geometric discrepancy or lack variations, further improving quality of the synthetic images. The experiments are conducted on both MS COCO and PASCAL VOC datasets using various state-of-the-art detectors (e.g., Faster R-CNN, RetinaNet, FCOS and Mask R-CNN). The results show that our proposed COCP can be well generalized to various datasets and detectors  with clear performance gains, while performing favorably against its counterparts.

```
@article{wang2020constrained,
  title={Constrained Online Cut-Paste for Object Detection},
  author={Wang, Hao and Wang, Qilong and Zhang, Hongzhi and Yang, Jian and Zuo, Wangmeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2020},
  publisher={IEEE}
}
```
