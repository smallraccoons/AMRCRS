## 重点
:star2: 所有数据的格式都转换成 lmdb 格式

## 下载
*  scene, web 和 document 数据集可在以下链接获得 [BaiduCloud](https://pan.baidu.com/s/1OlAAvSOUl8mA2WBzRC8RCg) (psw:v2rm) 或者 [GoogleDrive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing).

* 困难文本数据集可以从以下链接下载： [BaiduCloud](https://pan.baidu.com/s/1HjY_LuQPpBiol6Sc7noUDQ) (psw:n6nu) 或者 [GoogleDrive](https://drive.google.com/drive/folders/1J-3klWJasVJTL32FOKaFXZykKwN6Wni5?usp=sharing); the lmdb dataset for examples of synthetic CTR data is available in [BaiduCloud](https://pan.baidu.com/s/1ON3mwSJyXiWxZ00DxoHCxA) (psw:c4sl).

* 苦难识别样例可以从以下链接获得： [BaiduCloud](https://pan.baidu.com/s/1HjY_LuQPpBiol6Sc7noUDQ) (psw:n6nu).

* 对于手写数据集, 请从以下链接下载： [SCUT-HCCDoc](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release) 并将其按照 [link](https://github.com/smallraccoons/AMRCRS/tree/main/data)划分为 训练集,验证集和测试集 .

* 我们还从中国科学院自动化研究所[CASIA](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)和ICDAR2013竞赛中收集了HWDB2.0-2.2和ICDAR2013手写数据集，以供进一步研究使用。这些数据集可以在[BaiduCloud](https://pan.baidu.com/s/1q_x3L1lZBRykoY-AwhtoXw) （密码：lfaq）和[GoogleDrive](https://drive.google.com/drive/folders/1_xLYnEtoVo-RvPL9m79f0HgERwtR1Wc-?usp=sharing)上获取。
## 数据集
![Alt text](./images/dataset.png)
这张图片展示了我们实验中使用的四个数据集，包括场景、网络、文档和手写数据集，接下来将介绍每一个数据集。

# 声明 
我们实验研究的数据集，来自引用的benchmark[15]的数据集。非常感谢FudanVI实验室提供的统一数据集。


以下是四个数据集上基线模型的结果。ACC / NED分别采用百分比和小数格式。请点击超链接查看详细的实验结果，格式为（索引 [预测] [真实值]）。
<table><tbody>
    <tr>
        <th rowspan="2">&nbsp;&nbsp;Baseline&nbsp;&nbsp;</th>
        <th rowspan="2">&nbsp;&nbsp;Year&nbsp;&nbsp;</th>
        <th colspan="4">Dataset</th>
    </tr>
    <tr>
        <th align="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Scene&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th align="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Web&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
        <th align="center">&nbsp;&nbsp;Document&nbsp;&nbsp;</th>
        <th align="center">&nbsp;Handwriting&nbsp;</th>
    </tr>
    <tr>
        <td align="center">CRNN [9]</td>
        <td align="center">2016</td>
        <td align="center"><a href="./predictions/CRNN/CRNN_scene.txt" >54.94 / 0.742</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_web.txt" >56.21 / 0.745</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_document.txt">97.41 / 0.995</a></td>
        <td align="center"><a href="./predictions/CRNN/CRNN_handwriting.txt">48.04 / 0.843</a></td>
    </tr>
    <tr>
        <td align="center">ASTER [10]</td>
        <td align="center">2018</td>
        <td align="center"><a href="./predictions/ASTER/ASTER_scene.txt">59.37 / 0.801</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_web.txt">57.83 / 0.782</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_document.txt">97.59 / 0.995</a></td>
        <td align="center"><a href="./predictions/ASTER/ASTER_handwriting.txt">45.90 / 0.819</a></td>
    </tr>
    <tr>
        <td align="center">MORAN [11]</td>
        <td align="center">2019</td>
        <td align="center"><a href="./predictions/MORAN/MORAN_scene.txt">54.68 / 0.710</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_web.txt">49.64 / 0.679</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_document.txt">91.66 / 0.984</a></td>
        <td align="center"><a href="./predictions/MORAN/MORAN_handwriting.txt">30.24 / 0.651</a></td>
    </tr>
    <tr>
        <td align="center">SAR [12]</td>
        <td align="center">2019</td>
        <td align="center"><a href="./predictions/SAR/SAR_scene.txt">53.80 / 0.738</a></td>
        <td align="center"><a href="./predictions/SAR/SAR_web.txt">50.49 / 0.705</a></td>
        <td align="center"><a href="./predictions/SAR/SAR_document.txt">96.23 / 0.993</a></td>
        <td align="center"><a href="./predictions/SAR/SAR_handwriting.txt" >30.95 / 0.732</a></td>
    </tr>
    <tr>
        <td align="center">SEED [13]</td>
        <td align="center">2020</td>
        <td align="center"><a href="./predictions/SEED/SEED_scene.txt">45.37 / 0.708</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_web.txt">31.35 / 0.571</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_document.txt">96.08 / 0.992</a></td>
        <td align="center"><a href="./predictions/SEED/SEED_handwriting.txt">21.10 / 0.555</a></td>
    </tr>
    <tr>
        <td align="center">TransOCR [14]</td>
        <td align="center">2021</td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_scene.txt">67.81 / 0.817</a></td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_web.txt">62.74 / 0.782</a></td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_document.txt">97.86 / 0.996</a></td>
        <td align="center"><a href="./predictions/TransOCR/TransOCR_handwriting.txt">51.67 / 0.835</a></td>
    </tr>
    <tr>
        <td align="center">AMRCRS</td>
        <td align="center">Now</td>
        <td align="center"><a >72.64 / 0.857</a></td>
        <td align="center"><a >64.91 / 0.817</a></td>
        <td align="center"><a >98.17 / 0.997</a></td>
        <td align="center"><a >57.81 / 0.876</a></td>
    </tr>
</table>

## 参考文献

### 数据集
[1] Shi B, Yao C, Liao M, et al. ICDAR2017 competition on reading chinese text in the wild (RCTW-17). ICDAR, 2017. 

[2] Zhang R, Zhou Y, Jiang Q, et al. Icdar 2019 robust reading challenge on reading chinese text on signboard. ICDAR, 2019. 

[3] Sun Y, Ni Z, Chng C K, et al. ICDAR 2019 competition on large-scale street view text with partial labeling-RRC-LSVT. ICDAR, 2019. 

[4] Chng C K, Liu Y, Sun Y, et al. ICDAR2019 robust reading challenge on arbitrary-shaped text-RRC-ArT. ICDAR, 2019. 

[5] Yuan T L, Zhu Z, Xu K, et al. A large chinese text dataset in the wild. Journal of Computer Science and Technology, 2019.

[6] He M, Liu Y, Yang Z, et al. ICPR2018 contest on robust reading for multi-type web images. ICPR, 2018. 

[7] text_render: [https://github.com/Sanster/text_renderer](https://github.com/Sanster/text_renderer)

[8] Zhang H, Liang L, Jin L. SCUT-HCCDoc: A new benchmark dataset of handwritten Chinese text in unconstrained camera-captured documents. Pattern Recognition, 2020. 


### 方法
[9] Shi B, Bai X, Yao C. An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. TPAMI, 2016.

[10] Shi B, Yang M, Wang X, et al. Aster: An attentional scene text recognizer with flexible rectification. TPAMI, 2018.

[11] Luo C, Jin L, Sun Z. Moran: A multi-object rectified attention network for scene text recognition. PR, 2019.

[12] Li H, Wang P, Shen C, et al. Show, attend and read: A simple and strong baseline for irregular text recognition. AAAI, 2019.

[13] Qiao Z, Zhou Y, Yang D, et al. Seed: Semantics enhanced encoder-decoder framework for scene text recognition. CVPR, 2020.

[14] Chen J, Li B, Xue X. Scene Text Telescope: Text-Focused Scene Image Super-Resolution. CVPR, 2021.

[15] Yu H , Chen J , Li B ,et al. Benchmarking Chinese Text Recognition: Datasets, Baselines, and an Empirical Study[J].  2021.DOI:10.48550/arXiv.2112.15093.
