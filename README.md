
# [Infrared Image Super-Resolution: <br/> Systematic Review, and Future Trends](https://arxiv.org/pdf/2212.12322)

![](https://github.com/yongsongH/Infrared_Image_SR_Survey/blob/main/fig3.3.png)
---

- [1.Introduction](#1-introduction)
  - [1.1.Problem Definitions](#1-1-problem-definitions)
- [2.Fundamentals of Infrared Images](#2-fundamentals-of-infrared-images)
  - [2.1.Imaging System](#2-1-imaging-system)
    - [Difficulties with the Lense](#2-1-1-difficulties-with-the-lense)
    - [Difficulties with Sensors](#2-1-2-difficulties-with-sensors)
    - [Difficulties with Processors](#2-1-3-difficulties-with-processors)
  - [2.2 IR Image Synthesis Model](#2-2-ir-image-synthesis-model) 
  - [2.3 Applications](#2-2-applications)
     - [Medical biochemical engineering](#medical-biochemical-engineering)
        - [Pharmaceutical industry](#pharmaceutical-industry)
        - [Medical science](#medical-science) 
        - [Cellular observations](#cellular-observations) 
        - [Fluorescence microscopy](#fluorescence-microscopy) 
      - [Vision tasks](#vision-tasks)
        - [Image conversion](#image-conversion)
        - [Multispectral matching](#multispectral-matching)
        - [Targets detection](#targets-detection) 
        - [Face recognition](#face-recognition)
      - [Other engineering tasks](#other-engineering-tasks)
        - [Automated vehicle](#automated-vehicle) 
        - [Remote sensing](#remote-sensing) 
        - [Terrain models](#terrain-models)
        - [Land surface](#land-surface) 
        - [Food quality control](#food-quality-control) 
        - [Agricultural management](#agricultural-management) 
        - [Water resource management](#water-resource-management)
        - [Star formation](#star-formation) 

- [3.Related Methods](#4-related-methods)
     - 3.1.Traditional Methods
         - [Patches](#patches)
          - [Dictionary](#dictionary)
          - [Correspondence Relationship](#correspondence-relationship)
          - [Extra Information](#extra-information)
          - [Other methods](#other-methods)
          - [Iterative](#iterative)
          - [Sparsity](#sparsity)
          - [Sparse Representations](#sparse-representations)
          - [Projection](#projection)
          - [Regularization](#regularization)
         
     - [3.2.Deep learning](#3-2-deep-learning)
        - [CNN & Traditional Methods](#cnn-and-traditional-methods)
        - [CNN & End-to-end models](#cnn-and-end-to-end-models)
        - [GAN based](#gan-based)

- [4.Datasets & IQA](#4-datasets-and-iqa)
  - [Datasets](#datasets)
  - [IQA](#iqa)
      
- [5.Trends](#5-trends)
  - [Diffusion model](#diffusion-model)
  - [Transformer](#transformer)
  - [Blind SR](#blind-sr)

- End
---

## 1 Introduction


 Deep learning for image super-resolution: A survey

```
@article{wang2020deep,
  title={Deep learning for image super-resolution: A survey},
  author={Wang, Zhihao and Chen, Jian and Hoi, Steven CH},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={43},
  number={10},
  pages={3365--3387},
  year={2020},
  publisher={IEEE}
}
```

 Super-resolution image reconstruction: a technical overview
```
@article{park2003super,
  title={Super-resolution image reconstruction: a technical overview},
  author={Park, Sung Cheol and Park, Min Kyu and Kang, Moon Gi},
  journal={IEEE signal processing magazine},
  volume={20},
  number={3},
  pages={21--36},
  year={2003},
  publisher={IEEE}
}
```

Real-world single image super-resolution: A brief review
```
@article{chen2022real,
  title={Real-world single image super-resolution: A brief review},
  author={Chen, Honggang and He, Xiaohai and Qing, Linbo and Wu, 
  Yuanyuan and Ren, Chao and Sheriff, Ray E and Zhu, Ce},
  journal={Information Fusion},
  volume={79},
  pages={124--145},
  year={2022},
  publisher={Elsevier}
}
```
 Blind image super-resolution: A survey and beyond
```
@article{liu2022blind,
  title={Blind image super-resolution: A survey and beyond},
  author={Liu, Anran and Liu, Yihao and Gu, Jinjin and Qiao, Yu and Dong, Chao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
Image super-resolution using deep convolutional networks
```
@article{dong2015image,
  title={Image super-resolution using deep convolutional networks},
  author={Dong, Chao and Loy, Chen Change and He, Kaiming and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={2},
  pages={295--307},
  year={2015},
  publisher={IEEE}
}
```

Photo-realistic single image super-resolution using a generative adversarial network
```
@inproceedings{ledig2017photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, 
  Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, 
  Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes 
  and Wang, Zehan and others},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4681--4690},
  year={2017}
}
```

Esrgan: Enhanced super-resolution generative adversarial networks
```
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, 
  Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={Proceedings of the European conference on computer vision (ECCV) workshops},
  pages={0--0},
  year={2018}
}
```
Real-esrgan: Training real-world blind super-resolution with pure synthetic data
```
@inproceedings{wang2021real,
  title={Real-esrgan: Training real-world blind super-resolution with pure synthetic data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1905--1914},
  year={2021}
}
```

Swinir: Image restoration using swin transformer
```
@inproceedings{liang2021swinir,
  title={Swinir: Image restoration using swin transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei 
  and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1833--1844},
  year={2021}
}
```

A survey on vision transformer
```
@article{han2022survey,
  title={A survey on vision transformer},
  author={Han, Kai and Wang, Yunhe and Chen, Hanting and Chen, Xinghao and Guo,
  Jianyuan and Liu, Zhenhua and Tang, Yehui and Xiao, An and Xu, Chunjing and Xu, Yixing and others},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2022},
  publisher={IEEE}
}
```
Transformers in vision: A survey
```
@article{khan2021transformers,
  title={Transformers in vision: A survey},
  author={Khan, Salman and Naseer, Muzammal and Hayat, Munawar and Zamir, 
  Syed Waqas and Khan, Fahad Shahbaz and Shah, Mubarak},
  journal={ACM computing surveys (CSUR)},
  year={2021},
  publisher={ACM New York, NY}
}
```
An image is worth 16x16 words: Transformers for image recognition at scale
```
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, 
  Lucas and Kolesnikov, Alexander and Weissenborn, 
  Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, 
  Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

### 1-1 Problem Definitions
---
Image super-resolution as sparse representation of raw image patches
```
@inproceedings{yang2008image,
  title={Image super-resolution as sparse representation of raw image patches},
  author={Yang, Jianchao and Wright, John and Huang, Thomas and Ma, Yi},
  booktitle={2008 IEEE conference on computer vision and pattern recognition},
  pages={1--8},
  year={2008},
  organization={IEEE}
}
```

Image super-resolution via sparse representation
```
@article{yang2010image,
  title={Image super-resolution via sparse representation},
  author={Yang, Jianchao and Wright, John and Huang, Thomas S and Ma, Yi},
  journal={IEEE transactions on image processing},
  volume={19},
  number={11},
  pages={2861--2873},
  year={2010},
  publisher={IEEE}
}
```
Learning the degradation distribution for blind image super-resolution
```
@inproceedings{luo2022learning,
  title={Learning the degradation distribution for blind image super-resolution},
  author={Luo, Zhengxiong and Huang, Yan and Li, Shang and Wang, Liang and Tan, Tieniu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6063--6072},
  year={2022}
}
```

Designing a practical degradation model for deep blind image super-resolution
```
@inproceedings{zhang2021designing,
  title={Designing a practical degradation model for deep blind image super-resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4791--4800},
  year={2021}
}
```
## 2 Fundamentals of Infrared Images

## 2-1 Imaging Systems

### 2-1-1 Difficulties with the Lense

Lens design
```
@book{laikin2018lens,
  title={Lens design},
  author={Laikin, Milton},
  year={2018},
  publisher={CRC Press}
}
```
Compressive Hyperspectral Imaging and Super-resolution
```
@inproceedings{yuan2018compressive,
  title={Compressive Hyperspectral Imaging and Super-resolution},
  author={Yuan, Han and Yan, Fengxia and Chen, Xinmeng and Zhu, Jubo},
  booktitle={2018 IEEE 3rd International Conference on Image, Vision and Computing (ICIVC)},
  pages={618--623},
  year={2018},
  organization={IEEE}
}
```

Dynamic laser speckle and applications
```
@book{rabal2018dynamic,
  title={Dynamic laser speckle and applications},
  author={Rabal, Hector J and Braga Jr, Roberto A},
  year={2018},
  publisher={CRC press}
}
```
Image analysis and understanding using super resolution
```
@inproceedings{lee2007image,
  title={Image analysis and understanding using super resolution},
  author={Lee, Harry and Olson, Teresa and Manville, Drew and Cloud, Gene},
  booktitle={Display Technologies and Applications for Defense, Security, and Avionics},
  volume={6558},
  pages={95--101},
  year={2007},
  organization={SPIE}
}
```
Study of asymmetric or decentered multi-view designs for uncooled infrared imaging applications
```
@article{mas2020study,
  title={Study of asymmetric or decentered multi-view designs for uncooled infrared imaging applications},
  author={Mas, Adrien and Druart, Guillaume and De La Barri{\`e}re, Florence},
  journal={Optics Express},
  volume={28},
  number={23},
  pages={35216--35230},
  year={2020},
  publisher={Optical Society of America}
}
```
Effectiveness assessment of signal processing in the presence of smear
```
@article{bijl2012effectiveness,
  title={Effectiveness assessment of signal processing in the presence of smear},
  author={Bijl, Piet and Beintema, Jaap A and van der Leden, Natasja and Dijk, Judith},
  journal={Optical Engineering},
  volume={51},
  number={6},
  pages={063205},
  year={2012},
  publisher={SPIE}
}
```
### 2-1-2 Difficulties with Sensors

Multiblock compressed sensing imaging in real time
```
@article{李虎2023multiblock,
  title={Multiblock compressed sensing imaging in real time},
  author={李虎 and 刘雪峰 and 姚旭日 and 刘璠 and 窦申成 and 胡钛 and 翟光杰},
  journal={Journal of Infrared and Millimeter Waves},
  volume={42},
  number={1},
  pages={61--71},
  year={2023}
}

```

Averaging Current Adjustment Technique for Reducing Pixel Resistance Variation in a Bolometer-Type Uncooled Infrared Image Sensor
```
@article{kim2018averaging,
  title={Averaging Current Adjustment Technique for Reducing Pixel Resistance Variation in a Bolometer-Type Uncooled Infrared Image Sensor},
  author={Kim, Sang-Hwan and Choi, Byoung-Soo and Lee, Jimin and Lee, Junwoo and Park, Jae-Hyoun and Lee, Kyoung-Il and Shin, Jang-Kyoo},
  journal={Journal of Sensor Science and Technology},
  volume={27},
  number={6},
  pages={357--361},
  year={2018},
  publisher={The Korean Sensors Society}
}
```
Stripe noise removal for infrared image by minimizing difference between columns
```
@article{wang2016stripe,
  title={Stripe noise removal for infrared image by minimizing difference between columns},
  author={Wang, Shu-Peng},
  journal={Infrared Physics \& Technology},
  volume={77},
  pages={58--64},
  year={2016},
  publisher={Elsevier}
}
```
Super resolution infrared camera using single carbon nanotube photodetector
```
@inproceedings{chen2014super,
  title={Super resolution infrared camera using single carbon nanotube photodetector},
  author={Chen, Liangliang and Zhou, Zhanxin and Xi, Ning and Yang, Ruiguo and Song, Bo and Sun, Zhiyong and Su, Chengzhi},
  booktitle={SENSORS, 2014 IEEE},
  pages={1038--1041},
  year={2014},
  organization={IEEE}
}
```

Near-infrared super resolution imaging with metallic nanoshell particle chain array
```
@article{kong2013near,
  title={Near-infrared super resolution imaging with metallic nanoshell particle chain array},
  author={Kong, Weijie and Cao, Pengfei and Zhang, Xiaoping and Cheng, Lin and Wang, Tao and Yang, Lili and Meng, Qingqing},
  journal={Plasmonics},
  volume={8},
  number={2},
  pages={835--842},
  year={2013},
  publisher={Springer}
}
```
Signal conditioning algorithms for enhanced tactical sensor imagery
```
@inproceedings{schutte2003signal,
  title={Signal conditioning algorithms for enhanced tactical sensor imagery},
  author={Schutte, Klamer and de Lange, Dirk-Jan J and van den Broek, Sebastian P},
  booktitle={Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XIV},
  volume={5076},
  pages={92--100},
  year={2003},
  organization={SPIE}
}
```

Effect of pixel active area shapes in microscanning based infrared super-resolution imaging
```
@inproceedings{ming2013effect,
  title={Effect of pixel active area shapes in microscanning based infrared super-resolution imaging},
  author={Ming-Jie, Sun and Kang-long, Yu and Zhi, Xiao},
  booktitle={2013 Third International Conference on Instrumentation, Measurement, Computer, Communication and Control},
  pages={909--912},
  year={2013},
  organization={IEEE}
}
```
A sur-pixel scan method for super-resolution reconstruction
```
@article{sun2013pixel,
  title={A sur-pixel scan method for super-resolution reconstruction},
  author={Sun, Mingjie and Yu, Kanglong},
  journal={Optik},
  volume={124},
  number={24},
  pages={6905--6909},
  year={2013},
  publisher={Elsevier}
}
```

Sub-wavelength resolution of MMW imaging systems using extremely inexpensive scanning Glow Discharge Detector (GDD) double row camera
```
@inproceedings{kopeika2012sub,
  title={Sub-wavelength resolution of MMW imaging systems using extremely inexpensive scanning Glow Discharge Detector (GDD) double row camera},
  author={Kopeika, NS and Abramovich, A and Levanon, A and Akram, A and Rozban, D and Yitzhaky, Y and Yadid-Pecht, O and Belenky, A},
  booktitle={Passive and Active Millimeter-Wave Imaging XV},
  volume={8362},
  pages={127--134},
  year={2012},
  organization={SPIE}
}
```

### 2-1-3 Difficulties with Processors

Superresolution image reconstruction from a sequence of aliased imagery
```
@article{young2006superresolution,
  title={Superresolution image reconstruction from a sequence of aliased imagery},
  author={Young, S Susan and Driggers, Ronald G},
  journal={Applied Optics},
  volume={45},
  number={21},
  pages={5073--5085},
  year={2006},
  publisher={Optica Publishing Group}
}
```
### 2-2 IR Image Synthesis Model

An infrared image synthesis model based on infrared physics and heat transfer
```
@article{yu1998infrared,
  title={An infrared image synthesis model based on infrared physics and heat transfer},
  author={Yu, Weijie and Peng, Qunsheng and Tu, Hongming and Wang, Zhangye},
  journal={International journal of infrared and millimeter waves},
  volume={19},
  pages={1661--1669},
  year={1998},
  publisher={Springer}
}
```
Integrated modelling of thermal and visual image generation
```
@inproceedings{oh1989integrated,
  title={Integrated modelling of thermal and visual image generation},
  author={Oh, Chanhee and Nandhakumar, Nagaraj and Aggarwal, Jake K},
  booktitle={1989 IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
  pages={356--357},
  year={1989},
  organization={IEEE Computer Society}
}
```
Unified modeling of non-homogeneous 3D objects for thermal and visual image synthesis
```
@article{nandhakumar1994unified,
  title={Unified modeling of non-homogeneous 3D objects for thermal and visual image synthesis},
  author={Nandhakumar, Nagaraj and Karthik, Sankaran and Aggarwal, Jake K},
  journal={Pattern recognition},
  volume={27},
  number={10},
  pages={1303--1316},
  year={1994},
  publisher={Elsevier}
}
```
Simulation of reticle seekers using the generated thermal images
```
@inproceedings{hong1996simulation,
  title={Simulation of reticle seekers using the generated thermal images},
  author={Hong, Hyun-Ki and Han, Sung-Hyun and Hong, Gyoung-Pyo and Choi, Jong-Soo},
  booktitle={Proceedings of APCCAS'96-Asia Pacific Conference on Circuits and Systems},
  pages={183--186},
  year={1996},
  organization={IEEE}
}
```

### 2-3 Applications

### Medical biochemical engineering

### Pharmaceutical industry

Increasing the spatial resolution of near infrared chemical images (NIR-CI): The super-resolution paradigm applied to pharmaceutical products
```
@article{Offroy2012IncreasingTS,
  title={Increasing the spatial resolution of near infrared chemical images (NIR-CI): The super-resolution paradigm applied to pharmaceutical products},
  author={Marc Offroy and Yves Roggo and Ludovic Duponchel},
  journal={Chemometrics and Intelligent Laboratory Systems},
  year={2012},
  volume={117},
  pages={183-188}
}
```
Infrared chemical imaging: spatial resolution evaluation and super-resolution concept
```
@article{Offroy2010InfraredCI,
  title={Infrared chemical imaging: spatial resolution evaluation and super-resolution concept.},
  author={Marc Offroy and Yves Roggo and Peyman Milanfar and Ludovic Duponchel},
  journal={Analytica chimica acta},
  year={2010},
  volume={674 2},
  pages={
          220-6
        }
}
```

### Medical science 

Optical technologies for the detection of viruses like COVID-19: Progress and prospects
```
@article{Lukose2021OpticalTF,
  title={Optical technologies for the detection of viruses like COVID-19: Progress and prospects},
  author={Jijo Lukose and Santhosh Chidangil and Sajan D. George},
  journal={Biosensors \& Bioelectronics},
  year={2021},
  volume={178},
  pages={113004 - 113004}
}
```
COVID-19 classification using thermal images: thermal images capability for identifying COVID-19 using traditional machine learning classifiers
```
@article{CanalesFiscal2021COVID19CU,
  title={COVID-19 classification using thermal images: thermal images capability for identifying COVID-19 using traditional machine learning classifiers},
  author={Martha Rebeca Canales-Fiscal and Roc{\'i}o Ortiz L{\'o}pez and Regina Barzilay and V{\'i}ctor Trevi{\~n}o and Servando Cardona-Huerta and Luis Javier Ram{\'i}rez-Trevi{\~n}o and Adam Yala and Jose Gerardo Tamez-Pe{\~n}a},
  journal={Proceedings of the 12th ACM Conference on Bioinformatics, Computational Biology, and Health Informatics},
  year={2021}
}
```


Versatile near-infrared super-resolution imaging of amyloid fibrils with the fluorogenic probe CRANAD-2
```
@article{Torra2022VersatileNS,
  title={Versatile near-infrared super-resolution imaging of amyloid fibrils with the fluorogenic probe CRANAD-2.},
  author={Joaquim Torra and Felipe Viela and Diego Meg{\'i}as and Bego{\~n}a Sot and Cristina Flors},
  journal={Chemistry},
  year={2022}
}
```


Optical technologies for the detection of viruses like COVID-19: Progress and prospects
```
@article{lukose2021optical,
  title={Optical technologies for the detection of viruses like COVID-19: Progress and prospects},
  author={Lukose, Jijo and Chidangil, Santhosh and George, Sajan D},
  journal={Biosensors and Bioelectronics},
  volume={178},
  pages={113004},
  year={2021},
  publisher={Elsevier}
}
```

An infrared image denoising model with unidirectional gradient and sparsity constraint on biomedical images
```
@article{liu2022infrared,
  title={An infrared image denoising model with unidirectional gradient and sparsity constraint on biomedical images},
  author={Liu, Hai and An, Qing and Liu, Tingting and Huang, Zhenghua and Deng, Qian},
  journal={Infrared Physics \& Technology},
  volume={126},
  pages={104348},
  year={2022},
  publisher={Elsevier}
}
```

### Cellular observations

Live Cell Visualization of Multiple Protein-Protein Interactions with BiFC Rainbow
```
@article{Wang2018LiveCV,
  title={Live Cell Visualization of Multiple Protein-Protein Interactions with BiFC Rainbow.},
  author={Sheng Wang and Miao Ding and Boxin Xue and Yingping Hou and Yujie Sun},
  journal={ACS chemical biology},
  year={2018},
  volume={13 5},
  pages={
          1180-1188
        }
}
```

Repurposing a photosynthetic antenna protein as a super-resolution microscopy label
```
@article{Barnett2017RepurposingAP,
  title={Repurposing a photosynthetic antenna protein as a super-resolution microscopy label},
  author={Samuel F H Barnett and Andrew Hitchcock and Amit Kumar Mandal and Cvetelin Vasilev and Jonathan M Yuen and James Morby and Amanda A Brindley and Dariusz M. Niedzwiedzki and Donald A. Bryant and Ashley J. Cadby and Dewey Holten and Christopher Neil Hunter},
  journal={Scientific Reports},
  year={2017},
  volume={7}
}
```

Two-photon laser scanning fluorescence microscopy for functional cellular imaging: Advantages and challenges or One photon is good... but two is better!
```
@article{dufour2006two,
  title={Two-photon laser scanning fluorescence microscopy for functional cellular imaging: Advantages and challenges or One photon is good... but two is better!},
  author={Dufour, Pascal and Dufour, Suzie and Castonguay, Annie and McCarthy, Nathalie and De Koninck, Yves},
  journal={Medecine Sciences: M/S},
  volume={22},
  number={10},
  pages={837--844},
  year={2006}
}
```

### Fluorescence microscopy

Spontaneously blinking fluorophores based on nucleophilic addition/dissociation of intracellular glutathione for live-cell super-resolution imaging
```
@article{Morozumi2020SpontaneouslyBF,
  title={Spontaneously blinking fluorophores based on nucleophilic addition/dissociation of intracellular glutathione for live-cell super-resolution imaging.},
  author={Akihiko Morozumi and Mako Kamiya and Shin-Nosuke Uno and Keitaro Umezawa and Ryosuke Kojima and Toshitada Yoshihara and Seiji Tobita and Yasuteru Urano},
  journal={Journal of the American Chemical Society},
  year={2020}
}
```
Investigating the Performances of Wide-Field Raman Microscopy with Stochastic Optical Reconstruction Post-Processing
```
@article{mazaheri2022investigating,
  title={Investigating the Performances of Wide-Field Raman Microscopy with Stochastic Optical Reconstruction Post-Processing},
  author={Mazaheri, Leila and Jelken, Joachim and Avil{\'e}s, Mar{\'\i}a O and Legge, Sydney and Lagugn{\'e}-Labarthet, Fran{\c{c}}ois},
  journal={Applied spectroscopy},
  volume={76},
  number={3},
  pages={340--351},
  year={2022},
  publisher={SAGE Publications Sage UK: London, England}
}
```

Deep image restoration for infrared photothermal heterodyne imaging
```
@article{zhang2021deep,
  title={Deep image restoration for infrared photothermal heterodyne imaging},
  author={Zhang, Shuang and Kniazev, Kirill and Pavlovetc, Ilia M and Zhang, Shubin and Stevenson, Robert L and Kuno, Masaru},
  journal={The Journal of Chemical Physics},
  volume={155},
  number={21},
  pages={214202},
  year={2021},
  publisher={AIP Publishing LLC}
}
```
Quantitative infrared photothermal microscopy
```
@inproceedings{pavlovetc2020quantitative,
  title={Quantitative infrared photothermal microscopy},
  author={Pavlovetc, Ilia M and Podshivaylov, Eduard A and Frantsuzov, Pavel A and Hartland, Gregory V and Kuno, Masaru},
  booktitle={Single Molecule Spectroscopy and Superresolution Imaging XIII},
  volume={11246},
  pages={126--132},
  year={2020},
  organization={SPIE}
}
```

Super-resolution imaging with mid-IR photothermal microscopy on the single particle level
```
@inproceedings{li2015super,
  title={Super-resolution imaging with mid-IR photothermal microscopy on the single particle level},
  author={Li, Zhongming and Kuno, Masaru and Hartland, Gregory},
  booktitle={Physical Chemistry of Interfaces and Nanomaterials XIV},
  volume={9549},
  pages={121--128},
  year={2015},
  organization={SPIE}
}
```

Ultralong-Term Super-Resolution Tracking of Lysosomes in Brain Organoids by Near-Infrared Noble Metal Nanoclusters
```
@article{qiu2022ultralong,
  title={Ultralong-Term Super-Resolution Tracking of Lysosomes in Brain Organoids by Near-Infrared Noble Metal Nanoclusters},
  author={Qiu, Kangqiang and Yadav, Aditya and Tian, Zhiqi and Guo, Ziyuan and Shi, Donglu and Nandi, Chayan K and Diao, Jiajie},
  journal={ACS Materials Letters},
  volume={4},
  number={9},
  pages={1565--1573},
  year={2022},
  publisher={ACS Publications}
}
```

###  Vision tasks

### Image conversion

Super-resolution Thermal Generative Adversarial Networks for Infrared Image Enhancement
```
@article{Lee2022SuperresolutionTG,
  title={Super-resolution Thermal Generative Adversarial Networks for Infrared Image Enhancement},
  author={In Ho Lee and Won-Yeung Chung and Chan-Gook Park},
  journal={Journal of Institute of Control, Robotics and Systems},
  year={2022}
}
```
### Multispectral matching 

Multispectral Matching using Conditional Generative Appearance Modeling
```
@article{Bodensteiner2018MultispectralMU,
  title={Multispectral Matching using Conditional Generative Appearance Modeling},
  author={Christoph Bodensteiner and Sebastian Bullinger and Michael Arens},
  journal={2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
  year={2018},
  pages={1-6}
}
```


Multimodal Sensor Fusion In Single Thermal image Super-Resolution
```
@inproceedings{Almasri2018MultimodalSF,
  title={Multimodal Sensor Fusion In Single Thermal image Super-Resolution},
  author={Feras Almasri and Olivier Debeir},
  booktitle={ACCV Workshops},
  year={2018}
}
```

### Targets detection 

An infrared image detection of power equipment based on super-resolution reconstruction and YOLOv4
```
@article{wu2022infrared,
  title={An infrared image detection of power equipment based on super-resolution reconstruction and YOLOv4},
  author={Wu, Junpeng and Li, Xianglei and Zhou, Yibo},
  journal={The Journal of Engineering},
  volume={2022},
  number={10},
  pages={1006--1016},
  year={2022},
  publisher={Wiley Online Library}
}
```

YOLO-SASE: An Improved YOLO Algorithm for the Small Targets Detection in Complex Backgrounds
```
@article{Zhou2022YOLOSASEAI,
  title={YOLO-SASE: An Improved YOLO Algorithm for the Small Targets Detection in Complex Backgrounds},
  author={Xiao Zhou and Lang Jiang and Caixia Hu and Shuai Lei and Tingting Zhang and Xingang Mou},
  journal={Sensors (Basel, Switzerland)},
  year={2022},
  volume={22}
}
```

Improved YOLOv4 network using infrared images for personnel detection in coal mines
```
@article{Li2022ImprovedYN,
  title={Improved YOLOv4 network using infrared images for personnel detection in coal mines},
  author={Xiaoyu Li and Shuai Wang and B. Liu and Wei Chen and Weiqiang Fan and Zijian Tian},
  journal={Journal of Electronic Imaging},
  year={2022},
  volume={31},
  pages={013017 - 013017}
}
```
Embedded system for real-time person detecting in infrared images/videos using super-resolution and Haar-like feature techniques
```
@article{Ramos2015EmbeddedSF,
  title={Embedded system for real-time person detecting in infrared images/videos using super-resolution and Haar-like feature techniques},
  author={G. Ramos and Juan Carlos Garc{\'i}a and Volodymyr Ponomariov},
  journal={2015 12th International Conference on Electrical Engineering, Computing Science and Automatic Control (CCE)},
  year={2015},
  pages={1-6}
}
```

### Face recognition 

Influence of thermal imagery resolution on accuracy of deep learning based face recognition
```
@inproceedings{szankin2019influence,
  title={Influence of thermal imagery resolution on accuracy of deep learning based face recognition},
  author={Szankin, Maciej and Kwasniewska, Alicja and Ruminski, Jacek},
  booktitle={2019 12th International Conference on Human System Interaction (HSI)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```

A super-resolution based method to synthesize visual images from near infrared
```
@inproceedings{shao2009super,
  title={A super-resolution based method to synthesize visual images from near infrared},
  author={Shao, Ming and Wang, Yunhong and Wang, Yiding},
  booktitle={2009 16th IEEE International Conference on Image Processing (ICIP)},
  pages={2453--2456},
  year={2009},
  organization={IEEE}
}
```

Face synthesis from low-resolution near-infrared to high-resolution visual light spectrum based on tensor analysis
```
@article{zhang2014face,
  title={Face synthesis from low-resolution near-infrared to high-resolution visual light spectrum based on tensor analysis},
  author={Zhang, Zhaoxiang and Wang, Yunhong and Zhang, Zeda},
  journal={Neurocomputing},
  volume={140},
  pages={146--154},
  year={2014},
  publisher={Elsevier}
}
```

Edge Multidirectional Binary Pattern Applies to High Resolution Thermal Infrared Face Database
```
@inproceedings{zhang2015edge,
  title={Edge Multidirectional Binary Pattern Applies to High Resolution Thermal Infrared Face Database},
  author={Zhang, Xiaoyuan and Yang, Jucheng and Liu, Na and Liu, Jianzheng},
  booktitle={Chinese Conference on Biometric Recognition},
  pages={508--515},
  year={2015},
  organization={Springer}
}
```

An embedded hardware architecture for real-time super-resolution in infrared cameras
```
@inproceedings{redlich2016embedded,
  title={An embedded hardware architecture for real-time super-resolution in infrared cameras},
  author={Redlich, Rodolfo and Araneda, Luis and Saavedra, Antonio and Figueroa, Miguel},
  booktitle={2016 Euromicro conference on digital system design (DSD)},
  pages={184--191},
  year={2016},
  organization={IEEE}
}
```

### Other engineering tasks

4D Super-Resolution Microscopy with Conventional Fluorophores and Single Wavelength Excitation in Optically Thick Cells and Tissues
```
@article{Baddeley20114DSM,
  title={4D Super-Resolution Microscopy with Conventional Fluorophores and Single Wavelength Excitation in Optically Thick Cells and Tissues},
  author={David Baddeley and David J. Crossman and Sabrina Rossberger and Juliette E. Cheyne and Johanna M. Montgomery and Isuru D Jayasinghe and Christoph Cremer and Mark B. Cannell and Christian Soeller},
  journal={PLoS ONE},
  year={2011},
  volume={6}
}
```

Tumor Retention of Enzyme-Responsive Pt(II) Drug-Loaded Nanoparticles Imaged by Nanoscale Secondary Ion Mass Spectrometry and Fluorescence Microscopy
```
@article{Proetto2018TumorRO,
  title={Tumor Retention of Enzyme-Responsive Pt(II) Drug-Loaded Nanoparticles Imaged by Nanoscale Secondary Ion Mass Spectrometry and Fluorescence Microscopy},
  author={Maria T. Proetto and Cassandra E. Callmann and John B. Cliff and Craig Szymanski and Dehong Hu and Stephen B. Howell and James E. Evans and Galya Orr and Nathan C. Gianneschi},
  journal={ACS Central Science},
  year={2018},
  volume={4},
  pages={1477 - 1484}
}
```


When Super-Resolution Localization Microscopy Meets Carbon Nanotubes
```
@article{Nandi2022WhenSL,
  title={When Super-Resolution Localization Microscopy Meets Carbon Nanotubes},
  author={Somen Nandi and Karen Caicedo and Laurent Cognet},
  journal={Nanomaterials},
  year={2022},
  volume={12}
}
```
Optimization of Advanced Live-Cell Imaging through Red/Near-Infrared Dye Labeling and Fluorescence Lifetime-Based Strategies
```
@article{Benard2021OptimizationOA,
  title={Optimization of Advanced Live-Cell Imaging through Red/Near-Infrared Dye Labeling and Fluorescence Lifetime-Based Strategies},
  author={Magalie Beenard and Damien Schapman and others},
  journal={International Journal of Molecular Sciences},
  year={2021},
  volume={22}
}
```





YOLO-SASE: An Improved YOLO Algorithm for the Small Targets Detection in Complex Backgrounds
```
@article{zhou2022yolo,
  title={YOLO-SASE: An Improved YOLO Algorithm for the Small Targets Detection in Complex Backgrounds},
  author={Zhou, Xiao and Jiang, Lang and Hu, Caixia and Lei, Shuai and Zhang, Tingting and Mou, Xingang},
  journal={Sensors},
  volume={22},
  number={12},
  pages={4600},
  year={2022},
  publisher={MDPI}
}
```

Embedded system for real-time person detecting in infrared images/videos using super-resolution and Haar-like feature techniques
```
@inproceedings{ramos2015embedded,
  title={Embedded system for real-time person detecting in infrared images/videos using super-resolution and Haar-like feature techniques},
  author={Ramos, Gilberto Guadalupe Jara and Garcia, Juan Carlos Sanchez and Ponomariov, Volodymyr},
  booktitle={2015 12th International Conference on Electrical Engineering, Computing Science and Automatic Control (CCE)},
  pages={1--6},
  year={2015},
  organization={IEEE}
}
```

A depth estimation framework based on unsupervised learning and cross-modal translation
```
@inproceedings{shen2019depth,
  title={A depth estimation framework based on unsupervised learning and cross-modal translation},
  author={Shen, Jiafeng and Wang, Kaiwei and Yang, Kailun and Xiang, Kaite and Fei, Lei and Hu, Xinxin and Li, Huabing and Chen, Hao},
  booktitle={Target and Background Signatures V},
  volume={11158},
  pages={21--31},
  year={2019},
  organization={SPIE}
}
```

Image enhancement framework for low-resolution thermal images in visible and LWIR camera systems
```
@inproceedings{rukkanchanunt2017image,
  title={Image enhancement framework for low-resolution thermal images in visible and LWIR camera systems},
  author={Rukkanchanunt, Thapanapong and Tanaka, Masayuki and Okutomi, Masatoshi},
  booktitle={Emerging Imaging and Sensing Technologies for Security and Defence II},
  volume={10438},
  pages={37--46},
  year={2017},
  organization={SPIE}
}
```


### Automated vehicle 

Toward Unaligned Guided Thermal Super-Resolution
```
@article{Gupta2022TowardUG,
  title={Toward Unaligned Guided Thermal Super-Resolution},
  author={Honey Gupta and Kaushik Mitra},
  journal={IEEE Transactions on Image Processing},
  year={2022},
  volume={31},
  pages={433-445}
}
```
A depth estimation framework based on unsupervised learning and cross-modal translation
```
@inproceedings{Shen2019ADE,
  title={A depth estimation framework based on unsupervised learning and cross-modal translation},
  author={Jiafeng Shen and Kaiwei Wang and Kailun Yang and Kaite Xiang and Lei Fei and Xinxin Hu and Huabing Li and Hao Chen},
  booktitle={Security + Defence},
  year={2019}
}
```
Image enhancement framework for low-resolution thermal images in visible and LWIR camera systems
```
@inproceedings{Rukkanchanunt2017ImageEF,
  title={Image enhancement framework for low-resolution thermal images in visible and LWIR camera systems},
  author={Thapanapong Rukkanchanunt and Masayuki Tanaka and M. Okutomi},
  booktitle={Security + Defence},
  year={2017}
}
```

### Remote sensing 

Deriving non-cloud contaminated sentinel-2 images with RGB and near-infrared bands from sentinel-1 images based on a conditional generative adversarial network
```
@article{xiong2021deriving,
  title={Deriving non-cloud contaminated sentinel-2 images with RGB and near-infrared bands from sentinel-1 images based on a conditional generative adversarial network},
  author={Xiong, Quan and Di, Liping and Feng, Quanlong and Liu, Diyou and Liu, Wei and Zan, Xuli and Zhang, Lin and Zhu, Dehai and Liu, Zhe and Yao, Xiaochuang and others},
  journal={Remote Sensing},
  volume={13},
  number={8},
  pages={1512},
  year={2021},
  publisher={MDPI}
}
```

Land surface temperature and emissivity retrieval from nighttime middle-infrared and thermal-infrared sentinel-3 images
```
@article{nie2020land,
  title={Land surface temperature and emissivity retrieval from nighttime middle-infrared and thermal-infrared sentinel-3 images},
  author={Nie, Jing and Ren, Huazhong and Zheng, Yitong and Ghent, Darren and Tansey, Kevin},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={18},
  number={5},
  pages={915--919},
  year={2020},
  publisher={IEEE}
}
```



Rock slope monitoring using drone based multispectral and thermal images
```
@inproceedings{yaacob2020rock,
  title={Rock slope monitoring using drone based multispectral and thermal images},
  author={Yaacob, Muhammad Latifi Mohd and Rashid, Ahmad Safuan A and Ismail, Afiqah and Sa’ari, Radzuan and Mustaffar, Mushairry and Yusof, Norbazlan Mohd and Abd Rahaman, Norisam and others},
  booktitle={IOP Conference Series: Earth and Environmental Science},
  volume={540},
  number={1},
  pages={012024},
  year={2020},
  organization={IOP Publishing}
}
```

Spectral unmixing for thermal infrared multi-spectral airborne imagery over urban environments: day and night synergy
```
@article{granero2020spectral,
  title={Spectral unmixing for thermal infrared multi-spectral airborne imagery over urban environments: day and night synergy},
  author={Granero-Belinchon, Carlos and Michel, Aurelie and Achard, Veronique and Briottet, Xavier},
  journal={Remote Sensing},
  volume={12},
  number={11},
  pages={1871},
  year={2020},
  publisher={MDPI}
}
```

### Terrain models

A novel convolutional neural network architecture of multispectral remote sensing images for automatic material classification
```
@article{lin2021novel,
  title={A novel convolutional neural network architecture of multispectral remote sensing images for automatic material classification},
  author={Lin, Chuen-Horng and Wang, Ting-You},
  journal={Signal Processing: Image Communication},
  volume={97},
  pages={116329},
  year={2021},
  publisher={Elsevier}
}
```

### Land surface

Geometry and adjacency effects in urban land surface temperature retrieval from high-spatial-resolution thermal infrared images
```
@article{chen2021geometry,
  title={Geometry and adjacency effects in urban land surface temperature retrieval from high-spatial-resolution thermal infrared images},
  author={Chen, Shanshan and Ren, Huazhong and Ye, Xin and Dong, Jiaji and Zheng, Yitong},
  journal={Remote Sensing of Environment},
  volume={262},
  pages={112518},
  year={2021},
  publisher={Elsevier}
}
```

Time of day impact on mapping agricultural subsurface drainage systems with UAV thermal infrared imagery
```
@article{allred2021time,
  title={Time of day impact on mapping agricultural subsurface drainage systems with UAV thermal infrared imagery},
  author={Allred, Barry and Martinez, Luis and Fessehazion, Melake K and Rouse, Greg and Koganti, Triven and Freeland, Robert and Eash, Neal and Wishart, DeBonne and Featheringill, Robert},
  journal={Agricultural Water Management},
  volume={256},
  pages={107071},
  year={2021},
  publisher={Elsevier}
}
```

### Food quality control 

The advantage of multispectral images in fruit quality control for extra virgin olive oil production
```
@article{martinez2022advantage,
  title={The advantage of multispectral images in fruit quality control for extra virgin olive oil production},
  author={Mart{\'\i}nez Gila, Diego M and Navarro Soto, Javiera P and Satorres Mart{\'\i}nez, Silvia and G{\'o}mez Ortega, Juan and G{\'a}mez Garc{\'\i}a, Javier},
  journal={Food Analytical Methods},
  volume={15},
  number={1},
  pages={75--84},
  year={2022},
  publisher={Springer}
}
```

Identification of tomatoes with early decay using visible and near infrared hyperspectral imaging and image-spectrum merging technique
```
@article{wang2021identification,
  title={Identification of tomatoes with early decay using visible and near infrared hyperspectral imaging and image-spectrum merging technique},
  author={Wang, Huting and Hu, Rong and Zhang, Mengyun and Zhai, Zhiqiang and Zhang, Ruoyu},
  journal={Journal of Food Process Engineering},
  volume={44},
  number={4},
  pages={e13654},
  year={2021},
  publisher={Wiley Online Library}
}
```

### Agricultural management 

Monitoring of sugar beet growth indicators using wide-dynamic-range vegetation index (WDRVI) derived from UAV multispectral images
```
@article{cao2020monitoring,
  title={Monitoring of sugar beet growth indicators using wide-dynamic-range vegetation index (WDRVI) derived from UAV multispectral images},
  author={Cao, Yang and Li, Guo Long and Luo, Yuan Kai and Pan, Qi and Zhang, Shao Ying},
  journal={Computers and Electronics in Agriculture},
  volume={171},
  pages={105331},
  year={2020},
  publisher={Elsevier}
}
```

Simulating the Leaf Area Index of Rice from Multispectral Images
```
@article{liu2021simulating,
  title={Simulating the Leaf Area Index of Rice from Multispectral Images},
  author={Liu, Shenzhou and Zeng, Wenzhi and Wu, Lifeng and Lei, Guoqing and Chen, Haorui and Gaiser, Thomas and Srivastava, Amit Kumar},
  journal={Remote Sensing},
  volume={13},
  number={18},
  pages={3663},
  year={2021},
  publisher={MDPI}
}
```

Estimation of peanut leaf area index from unmanned aerial vehicle multispectral images
```
@article{qi2020estimation,
  title={Estimation of peanut leaf area index from unmanned aerial vehicle multispectral images},
  author={Qi, Haixia and Zhu, Bingyu and Wu, Zeyu and Liang, Yu and Li, Jianwen and Wang, Leidi and Chen, Tingting and Lan, Yubin and Zhang, Lei},
  journal={Sensors},
  volume={20},
  number={23},
  pages={6732},
  year={2020},
  publisher={MDPI}
}
```
Comparison of Machine Learning Methods for Leaf Nitrogen Estimation in Corn Using Multispectral UAV images
```
@article{barzin2021comparison,
  title={Comparison of Machine Learning Methods for Leaf Nitrogen Estimation in Corn Using Multispectral UAV images},
  author={Barzin, Razieh and Kamangir, Hamid and Bora, Ganesh C},
  journal={Transactions of the ASABE},
  volume={64},
  number={6},
  pages={2089--2101},
  year={2021},
  publisher={American Society of Agricultural and Biological Engineers}
}
```
Water content detection of maize leaves based on multispectral images
```
@article{yao2020water,
  title={Water content detection of maize leaves based on multispectral images},
  author={Yao-qi, Peng and Ying-xin, Xiao and Ze-tian, Fu and Yu-hong, Dong and Xin-xing, Li and Hai-jun, Yan and Yong-jun, Zheng},
  journal={Spectroscopy and Spectral Analysis},
  volume={40},
  number={4},
  pages={1257--1262},
  year={2020},
  publisher={OFFICE SPECTROSCOPY \& SPECTRAL ANALYSIS NO 76 COLLAGE SOUTH RD BEIJING~…}
}
```

Segmentation of thermal infrared images of cucumber leaves using K-means clustering for estimating leaf wetness duration
```
@article{wen2020segmentation,
  title={Segmentation of thermal infrared images of cucumber leaves using K-means clustering for estimating leaf wetness duration},
  author={Wen, Dongmei and Ren, Aixin and Ji, Tao and Flores-Parra, Isabel Maria and Yang, Xinting and Li, Ming},
  journal={International Journal of Agricultural and Biological Engineering},
  volume={13},
  number={3},
  pages={161--167},
  year={2020}
}
```

Vine disease detection in UAV multispectral images using optimized image registration and deep learning segmentation approach
```
@article{kerkech2020vine,
  title={Vine disease detection in UAV multispectral images using optimized image registration and deep learning segmentation approach},
  author={Kerkech, Mohamed and Hafiane, Adel and Canals, Raphael},
  journal={Computers and Electronics in Agriculture},
  volume={174},
  pages={105446},
  year={2020},
  publisher={Elsevier}
}
```

### Water resource management 

Optically Enhanced Super-Resolution of Sea Surface Temperature Using Deep Learning
```
@article{lloyd2021optically,
  title={Optically Enhanced Super-Resolution of Sea Surface Temperature Using Deep Learning},
  author={Lloyd, David T and Abela, Aaron and Farrugia, Reuben A and Galea, Anthony and Valentino, Gianluca},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--14},
  year={2021},
  publisher={IEEE}
}
```

Can the Structure Similarity of Training Patches Affect the Sea Surface Temperature Deep Learning Super-Resolution?
```
@article{ping2021can,
  title={Can the Structure Similarity of Training Patches Affect the Sea Surface Temperature Deep Learning Super-Resolution?},
  author={Ping, Bo and Meng, Yunshan and Xue, Cunjin and Su, Fenzhen},
  journal={Remote Sensing},
  volume={13},
  number={18},
  pages={3568},
  year={2021},
  publisher={MDPI}
}
```

Water body extraction from Sentinel-3 image with multiscale spatiotemporal super-resolution mapping
```
@article{yang2020water,
  title={Water body extraction from Sentinel-3 image with multiscale spatiotemporal super-resolution mapping},
  author={Yang, Xiaohong and Li, Yue and Wei, Yu and Chen, Zhanlong and Xie, Peng},
  journal={Water},
  volume={12},
  number={9},
  pages={2605},
  year={2020},
  publisher={MDPI}
}
```

### Star formation 

Super-resolution Imaging of the Protoplanetary Disk HD 142527 Using Sparse Modeling
```
@article{yamaguchi2020super,
  title={Super-resolution Imaging of the Protoplanetary Disk HD 142527 Using Sparse Modeling},
  author={Yamaguchi, Masayuki and Akiyama, Kazunori and Tsukagoshi, Takashi and Muto, Takayuki and Kataoka, Akimasa and Tazaki, Fumie and Ikeda, Shiro and Fukagawa, Misato and Honma, Mareki and Kawabe, Ryohei},
  journal={The Astrophysical Journal},
  volume={895},
  number={2},
  pages={84},
  year={2020},
  publisher={IOP Publishing}
}
```
High-resolution mid-infrared imaging of the asymptotic giant branch star rv bootis with the steward observatory adaptive optics system
```
@article{biller2005high,
  title={High-resolution mid-infrared imaging of the asymptotic giant branch star rv bootis with the steward observatory adaptive optics system},
  author={Biller, BA and Close, LM and Li, A and Bieging, JH and Hoffmann, WF and Hinz, PM and Miller, D and Brusa, G and Lloyd-Hart, M and Wildi, F and others},
  journal={The Astrophysical Journal},
  volume={620},
  number={1},
  pages={450},
  year={2005},
  publisher={IOP Publishing}
}
```

Evidence for ongoing star formation in the Carina nebula.
```
@article{megeath1996evidence,
  title={Evidence for ongoing star formation in the Carina nebula.},
  author={Megeath, ST and Cox, P and Bronfman, L and Roelfsema, PR},
  journal={Astronomy and Astrophysics},
  volume={305},
  pages={296},
  year={1996}
}
```

Super-resolution Imaging of the Protoplanetary Disk HD 142527 Using Sparse Modeling
```
@article{Yamaguchi2020SuperresolutionIO,
  title={Super-resolution Imaging of the Protoplanetary Disk HD 142527 Using Sparse Modeling},
  author={Masayuki Yamaguchi and Kazunori Akiyama and Takashi Tsukagoshi and Takayuki Muto and Akimasa Kataoka and Fumie Tazaki and Shiro Ikeda and Misato Fukagawa and Mareki Honma and Ryohei Kawabe},
  journal={The Astrophysical Journal},
  year={2020},
  volume={895}
}
```
irst science results from SOFIA/forcast: Super-resolution imaging of the S140 cluster at 37 $\mu$m
```
@article{harvey2012first,
  title={First science results from SOFIA/forcast: Super-resolution imaging of the S140 cluster at 37 $\mu$m},
  author={Harvey, Paul M and Adams, Joseph D and Herter, Terry L and Gull, George and Schoenwald, Justin and Keller, Luke D and De Buizer, James M and Vacca, William and Reach, William and Becklin, EE},
  journal={The Astrophysical Journal Letters},
  volume={749},
  number={2},
  pages={L20},
  year={2012},
  publisher={IOP Publishing}
}
```
High-Resolution Mid-Infrared Imaging of the Asymptotic Giant Branch Star RV Bootis with the Steward Observatory Adaptive Optics System
```
@article{Biller2005HighResolutionMI,
  title={High-Resolution Mid-Infrared Imaging of the Asymptotic Giant Branch Star RV Bootis with the Steward Observatory Adaptive Optics System},
  author={Beth A. Biller and Laird M. Close and A. Li and John H. Bieging and William F. Hoffmann and Philip M. Hinz and D T Miller and Guido Brusa and Michael Lloyd-Hart and F. Wildi and Daniel Edward Potter and Benjamin D. Oppenheimer},
  journal={The Astrophysical Journal},
  year={2005},
  volume={620},
  pages={450 - 458}
}
```
Evidence for ongoing star formation in the Carina nebula
```
@article{Megeath1996EvidenceFO,
  title={Evidence for ongoing star formation in the Carina nebula.},
  author={S. Thomas Megeath and Pierre Cox and Leonardo Bronfman and Peter R. Roelfsema},
  journal={Astronomy and Astrophysics},
  year={1996},
  volume={305},
  pages={296-307}
}
```


# 4 Related Methods


An analysis of a robust super resolution algorithm for infrared imaging
```
@inproceedings{wang2009analysis,
  title={An analysis of a robust super resolution algorithm for infrared imaging},
  author={Wang, Jing and Ralph, Jason F and Goulermas, John Y},
  booktitle={2009 Proceedings of 6th International Symposium on Image and Signal Processing and Analysis},
  pages={158--163},
  year={2009},
  organization={IEEE}
}
```
Resolution improvement of infrared images using visible image information
```
@article{choi2011resolution,
  title={Resolution improvement of infrared images using visible image information},
  author={Choi, Kyuha and Kim, Changhyun and Kang, Myung-Ho and Ra, Jong Beom},
  journal={IEEE signal processing letters},
  volume={18},
  number={10},
  pages={611--614},
  year={2011},
  publisher={IEEE}
}
```
An infrared image super-resolution reconstruction method based on compressive sensing
```
@article{mao2016infrared,
  title={An infrared image super-resolution reconstruction method based on compressive sensing},
  author={Mao, Yuxing and Wang, Yan and Zhou, Jintao and Jia, Haiwei},
  journal={Infrared Physics \& Technology},
  volume={76},
  pages={735--739},
  year={2016},
  publisher={Elsevier}
}
```




Super-resolution through neighbor embedding
```
@inproceedings{chang2004super,
  title={Super-resolution through neighbor embedding},
  author={Chang, Hong and Yeung, Dit-Yan and Xiong, Yimin},
  booktitle={Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004.},
  volume={1},
  pages={I--I},
  year={2004},
  organization={IEEE}
}
```


### Patches

Infrared image super-resolution via locality-constrained group sparse model
```
@article{cheng2014infrared,
  title={Infrared image super-resolution via locality-constrained group sparse model},
  author={Cheng-Zhi, Deng and Wei, Tian and Pan, Chen and Sheng-Qian, Wang and Hua-Sheng, Zhu and Sai-Feng, Hu},
  journal={Acta Physica Sinica},
  volume={63},
  number={4},
  year={2014},
  publisher={CHINESE PHYSICAL SOC PO BOX 603, BEIJING 100080, PEOPLES R CHINA}
}
```
Infrared Image Recovery from Visible Image by Using Multi-scale and Multi-view Sparse Representation
```
@inproceedings{yang2015infrared,
  title={Infrared Image Recovery from Visible Image by Using Multi-scale and Multi-view Sparse Representation},
  author={Yang, Xiaomin and Wu, Wei and Hua, Hua and Liu, Kai},
  booktitle={2015 11th International Conference on Signal-Image Technology \& Internet-Based Systems (SITIS)},
  pages={554--559},
  year={2015},
  organization={IEEE}
}
```

Multi-sensor image super-resolution with fuzzy cluster by using multi-scale and multi-view sparse coding for infrared image
```
@article{yang2017multi,
  title={Multi-sensor image super-resolution with fuzzy cluster by using multi-scale and multi-view sparse coding for infrared image},
  author={Yang, Xiaomin and Wu, Wei and Liu, Kai and Chen, Weilong and Zhang, Ping and Zhou, Zhili},
  journal={Multimedia Tools and Applications},
  volume={76},
  number={23},
  pages={24871--24902},
  year={2017},
  publisher={Springer}
}
```

### Dictionary 

Fast multisensor infrared image super-resolution scheme with multiple regression models
```
@article{yang2016fast,
  title={Fast multisensor infrared image super-resolution scheme with multiple regression models},
  author={Yang, Xiaomin and Wu, Wei and Liu, Kai and Zhou, Kai and Yan, Binyu},
  journal={Journal of Systems Architecture},
  volume={64},
  pages={11--25},
  year={2016},
  publisher={Elsevier}
}
```
Infrared image super-resolution via discriminative dictionary and deep residual network
```
@article{yao2020infrared,
  title={Infrared image super-resolution via discriminative dictionary and deep residual network},
  author={Yao, Tingting and Luo, Yu and Hu, Jincheng and Xie, Haibo and Hu, Qing},
  journal={Infrared Physics \& Technology},
  volume={107},
  pages={103314},
  year={2020},
  publisher={Elsevier}
}
```

### Correspondence Relationship

A super-resolution reconstruction algorithm of infrared pedestrian images via compressed sensing
```
@inproceedings{zou2018super,
  title={A super-resolution reconstruction algorithm of infrared pedestrian images via compressed sensing},
  author={Zou, Erbo and Lei, Bo and Jing, Nan and Tan, Hai},
  booktitle={Real-time Photonic Measurements, Data Management, and Processing III},
  volume={10822},
  pages={106--111},
  year={2018},
  organization={SPIE}
}
```

### Extra Information

Multimodal image super-resolution via joint sparse representations induced by coupled dictionaries
```
@article{song2019multimodal,
  title={Multimodal image super-resolution via joint sparse representations induced by coupled dictionaries},
  author={Song, Pingfan and Deng, Xin and Mota, Joao FC and Deligiannis, Nikos and Dragotti, Pier Luigi and Rodrigues, Miguel RD},
  journal={IEEE Transactions on Computational Imaging},
  volume={6},
  pages={57--72},
  year={2019},
  publisher={IEEE}
}
```
Infrared image super-resolution reconstruction via sparse representation
```
@inproceedings{chen2018infrared,
  title={Infrared image super-resolution reconstruction via sparse representation},
  author={Chen, Zuming and Guo, Baolong and Zhang, Qi and Li, Cheng},
  booktitle={Journal of Physics: Conference Series},
  volume={1069},
  number={1},
  pages={012171},
  year={2018},
  organization={IOP Publishing}
}
```
Research on Blind Super-Resolution Technology for Infrared Images of Power Equipment Based on Compressed Sensing Theory
```
@article{wang2021research,
  title={Research on Blind Super-Resolution Technology for Infrared Images of Power Equipment Based on Compressed Sensing Theory},
  author={Wang, Yan and Wang, Lingjie and Liu, Bingcong and Zhao, Hongshan},
  journal={Sensors},
  volume={21},
  number={12},
  pages={4109},
  year={2021},
  publisher={MDPI}
}
```

### other methods.

### Iterative

Iris super-resolution using iterative neighbor embedding
```
@inproceedings{alonso2017iris,
  title={Iris super-resolution using iterative neighbor embedding},
  author={Alonso-Fernandez, Fernando and Farrugia, Reuben A and Bigun, Josef},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={153--161},
  year={2017}
}
```
Super resolution laser line scanning thermography
```
@article{ahmadi2020super,
  title={Super resolution laser line scanning thermography},
  author={Ahmadi, Samim and Burgholzer, P and Jung, P and Caire, G and Ziegler, Mathias},
  journal={Optics and Lasers in Engineering},
  volume={134},
  pages={106279},
  year={2020},
  publisher={Elsevier}
}
```

### Sparsity

Infrared image super-resolution reconstruction based on quaternion and high-order overlapping group sparse total variation
```
@article{liu2019infrared,
  title={Infrared image super-resolution reconstruction based on quaternion and high-order overlapping group sparse total variation},
  author={Liu, Xingguo and Chen, Yingpin and Peng, Zhenming and Wu, Juan},
  journal={Sensors},
  volume={19},
  number={23},
  pages={5139},
  year={2019},
  publisher={MDPI}
}
```
A Transform Learning Based Deconvolution Technique with Super-Resolution and Microscanning Applications
```
@inproceedings{gungor2019transform,
  title={A Transform Learning Based Deconvolution Technique with Super-Resolution and Microscanning Applications},
  author={G{\"u}ng{\"o}r, Alper and Kar, O{\u{g}}uzhan Fatih},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={2159--2163},
  year={2019},
  organization={IEEE}
}
```
Infrared image super-resolution reconstruction based on quaternion fractional order total variation with Lp quasinorm
```
@article{liu2018infrared,
  title={Infrared image super-resolution reconstruction based on quaternion fractional order total variation with Lp quasinorm},
  author={Liu, Xingguo and Chen, Yingpin and Peng, Zhenming and Wu, Juan and Wang, Zhuoran},
  journal={Applied Sciences},
  volume={8},
  number={10},
  pages={1864},
  year={2018},
  publisher={MDPI}
}
```
Research on super-resolution reconstruction algorithm of infrared images of compressive coded aperture
```
@inproceedings{chen2019research,
  title={Research on super-resolution reconstruction algorithm of infrared images of compressive coded aperture},
  author={Chen, Shaojun and Fan, Guihua and Zhang, Tinghua and Liu, Di},
  booktitle={Second Symposium on Novel Technology of X-Ray Imaging},
  volume={11068},
  pages={514--520},
  year={2019},
  organization={SPIE}
}
```

### Sparse Representations

Land cover target mapping at subpixel scale for Landsat 8 OLI image by using multiscale-infrared information
```
@article{wang2021land,
  title={Land cover target mapping at subpixel scale for Landsat 8 OLI image by using multiscale-infrared information},
  author={Wang, Peng and Yao, Hongyu and Zhang, Gong and Kong, Yingying and Lu, Shifang and Peng, Xiangyang},
  journal={International Journal of Remote Sensing},
  volume={42},
  number={3},
  pages={1054--1076},
  year={2021},
  publisher={Taylor \& Francis}
}
```

### Projection

Super-resolution imaging for infrared micro-scanning optical system
```
@article{zhang2019super,
  title={Super-resolution imaging for infrared micro-scanning optical system},
  author={Zhang, XF and Huang, W and Xu, MF and Jia, SQ and Xu, XR and Li, FB and Zheng, YD},
  journal={Optics express},
  volume={27},
  number={5},
  pages={7719--7737},
  year={2019},
  publisher={Optica Publishing Group}
}
```
An improved POCS super-resolution infrared image reconstruction algorithm based on visual mechanism
```
@article{liu2016improved,
  title={An improved POCS super-resolution infrared image reconstruction algorithm based on visual mechanism},
  author={Liu, Jinsong and Dai, Shaosheng and Guo, Zhongyuan and Zhang, Dezhou},
  journal={Infrared Physics \& Technology},
  volume={78},
  pages={92--98},
  year={2016},
  publisher={Elsevier}
}
```

### Regularization

Single infrared image super-resolution combining non-local means with kernel regression
```
@article{yu2013single,
  title={Single infrared image super-resolution combining non-local means with kernel regression},
  author={Yu, Hui and Chen, Fu-sheng and Zhang, Zhi-jie and Wang, Chen-sheng},
  journal={Infrared Physics \& Technology},
  volume={61},
  pages={50--59},
  year={2013},
  publisher={Elsevier}
}
```
A novel regularized adaptive edge-preserving image super\~{}-resolution algorithm
```
@article{hui2014novel,
  title={A novel regularized adaptive edge-preserving image super\~{}-resolution algorithm},
  author={Hui, Yu and Fu-Sheng, Cheng and Zhijie, Zhang and Chen-Sheng, Wang},
  journal={Infrared Millim Waves},
  volume={1},
  pages={98--105},
  year={2014}
}
```
Compressed Sensing Super-Resolution Method for Improving the Accuracy of Infrared Diagnosis of Power Equipment
```
@article{wang2022compressed,
  title={Compressed Sensing Super-Resolution Method for Improving the Accuracy of Infrared Diagnosis of Power Equipment},
  author={Wang, Yan and Zhang, Jialin and Wang, Lingjie},
  journal={Applied Sciences},
  volume={12},
  number={8},
  pages={4046},
  year={2022},
  publisher={MDPI}
}
```
Performance study on point target detection using super-resolution reconstruction
```
@inproceedings{dijk2009performance,
  title={Performance study on point target detection using super-resolution reconstruction},
  author={Dijk, Judith and van Eekeren, Adam WM and Schutte, Klamer and de Lange, Dirk-Jan J and van Vliet, Lucas J},
  booktitle={Automatic Target Recognition XIX},
  volume={7335},
  pages={202--209},
  year={2009},
  organization={SPIE}
}
```

## 4-2 Deep learning

Accelerating the super-resolution convolutional neural network
```
@inproceedings{dong2016accelerating,
  title={Accelerating the super-resolution convolutional neural network},
  author={Dong, Chao and Loy, Chen Change and Tang, Xiaoou},
  booktitle={European conference on computer vision},
  pages={391--407},
  year={2016},
  organization={Springer}
}
```
Enhanced deep residual networks for single image super-resolution
```
@inproceedings{lim2017enhanced,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}
```
Accurate image super-resolution using very deep convolutional networks
```
@inproceedings{kim2016accurate,
  title={Accurate image super-resolution using very deep convolutional networks},
  author={Kim, Jiwon and Lee, Jung Kwon and Lee, Kyoung Mu},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1646--1654},
  year={2016}
}
```

### CNN and Traditional Methods

Infrared image super resolution by combining compressive sensing and deep learning
```
@article{zhang2018infrared,
  title={Infrared image super resolution by combining compressive sensing and deep learning},
  author={Zhang, Xudong and Li, Chunlai and Meng, Qingpeng and Liu, Shijie and Zhang, Yue and Wang, Jianyu},
  journal={Sensors},
  volume={18},
  number={8},
  pages={2587},
  year={2018},
  publisher={MDPI}
}
```
Joint image super-resolution via recurrent convolutional neural networks with coupled sparse priors
```
@inproceedings{marivani2020joint,
  title={Joint image super-resolution via recurrent convolutional neural networks with coupled sparse priors},
  author={Marivani, Iman and Tsiligianni, Evaggelia and Cornelis, Bruno and Deligiannis, Nikos},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={868--872},
  year={2020},
  organization={IEEE}
}
```
Deep networks with detail enhancement for infrared image super-resolution
```
@article{yang2020deep,
  title={Deep networks with detail enhancement for infrared image super-resolution},
  author={Yang, Yifan and Li, Qi and Yang, Chenwei and Fu, Yannian and Feng, Huajun and Xu, Zhihai and Chen, Yueting},
  journal={IEEE Access},
  volume={8},
  pages={158690--158701},
  year={2020},
  publisher={IEEE}
}
```

### CNN and End-to-end models

Super-resolution reconstruction of infrared images based on a convolutional neural network with skip connections
```
@article{zou2021super,
  title={Super-resolution reconstruction of infrared images based on a convolutional neural network with skip connections},
  author={Zou, Yan and Zhang, Linfei and Liu, Chengqian and Wang, Bowen and Hu, Yan and Chen, Qian},
  journal={Optics and Lasers in Engineering},
  volume={146},
  pages={106717},
  year={2021},
  publisher={Elsevier}
}
```
Infrared Image Super-Resolution via Progressive Compact Distillation Network
```
@article{fan2021infrared,
  title={Infrared Image Super-Resolution via Progressive Compact Distillation Network},
  author={Fan, Kefeng and Hong, Kai and Li, Fei},
  journal={Electronics},
  volume={10},
  number={24},
  pages={3107},
  year={2021},
  publisher={MDPI}
}
```
Infrared image super-resolution via transfer learning and PSRGAN
```
@article{huang2021infrared,
  title={Infrared image super-resolution via transfer learning and PSRGAN},
  author={Huang, Yongsong and Jiang, Zetao and Lan, Rushi and Zhang, Shaoqin and Pi, Kui},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={982--986},
  year={2021},
  publisher={IEEE}
}
```
Channel split convolutional neural network (ChaSNet) for thermal image super-resolution
```
@inproceedings{prajapati2021channel,
  title={Channel split convolutional neural network (ChaSNet) for thermal image super-resolution},
  author={Prajapati, Kalpesh and Chudasama, Vishal and Patel, Heena and Sarvaiya, Anjali and Upla, Kishor P and Raja, Kiran and Ramachandra, Raghavendra and Busch, Christoph},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4368--4377},
  year={2021}
}
```
Super-resolved thermal imagery for high-accuracy facial areas detection and analysis
```
@article{kwasniewska2020super,
  title={Super-resolved thermal imagery for high-accuracy facial areas detection and analysis},
  author={Kwasniewska, Alicja and Ruminski, Jacek and Szankin, Maciej and Kaczmarek, Mariusz},
  journal={Engineering Applications of Artificial Intelligence},
  volume={87},
  pages={103263},
  year={2020},
  publisher={Elsevier}
}
```
Maritime Infrared Image Super-Resolution Using Cascaded Residual Network and Novel Evaluation Metric
```
@article{gao2022maritime,
  title={Maritime Infrared Image Super-Resolution Using Cascaded Residual Network and Novel Evaluation Metric},
  author={Gao, Zongjiang and Chen, Jinhai},
  journal={IEEE Access},
  volume={10},
  pages={17760--17767},
  year={2022},
  publisher={IEEE}
}
```
Thermal image superresolution through deep convolutional neural network
```
@inproceedings{rivadeneira2019thermal,
  title={Thermal image superresolution through deep convolutional neural network},
  author={Rivadeneira, Rafael E and Su{\'a}rez, Patricia L and Sappa, Angel D and Vintimilla, Boris X},
  booktitle={International conference on image analysis and recognition},
  pages={417--426},
  year={2019},
  organization={Springer}
}
```
ThermISRnet: an efficient thermal image super-resolution network
```
@article{patel2021thermisrnet,
  title={ThermISRnet: an efficient thermal image super-resolution network},
  author={Patel, Heena M and Chudasama, Vishal M and Prajapati, Kalpesh and Upla, Kishor P and Raja, Kiran and Ramachandra, Raghavendra and Busch, Christoph},
  journal={Optical Engineering},
  volume={60},
  number={7},
  pages={073101},
  year={2021},
  publisher={International Society for Optics and Photonics}
}
```

An infrared image super-resolution imaging algorithm based on auxiliary convolution neural network
```
@inproceedings{zou2020infrared,
  title={An infrared image super-resolution imaging algorithm based on auxiliary convolution neural network},
  author={Zou, Yan and Zhang, Linfei and Chen, Qian and Wang, Bowen and Hu, Yan and Zhang, Yuzhen},
  booktitle={Optics Frontier Online 2020: Optics Imaging and Display},
  volume={11571},
  pages={335--340},
  year={2020},
  organization={SPIE}
}
```

Rapid super resolution for infrared imagery
```
@article{oz2020rapid,
  title={Rapid super resolution for infrared imagery},
  author={Oz, Navot and Sochen, Nir and Markovich, Oshry and Halamish, Ziv and Shpialter-Karol, Lena and Klapp, Iftach},
  journal={Optics Express},
  volume={28},
  number={18},
  pages={27196--27209},
  year={2020},
  publisher={Optical Society of America}
}
```

Image fusion and super-resolution with convolutional neural network
```
@inproceedings{zhong2016image,
  title={Image fusion and super-resolution with convolutional neural network},
  author={Zhong, Jinying and Yang, Bin and Li, Yuehua and Zhong, Fei and Chen, Zhongze},
  booktitle={Chinese Conference on Pattern Recognition},
  pages={78--88},
  year={2016},
  organization={Springer}
}
```


Multimodal super-resolution reconstruction of infrared and visible images via deep learning
```
@article{wang2022multimodal,
  title={Multimodal super-resolution reconstruction of infrared and visible images via deep learning},
  author={Wang, Bowen and Zou, Yan and Zhang, Linfei and Li, Yuhai and Chen, Qian and Zuo, Chao},
  journal={Optics and Lasers in Engineering},
  volume={156},
  pages={107078},
  year={2022},
  publisher={Elsevier}
}
```

Infrared and visible light dual-camera super-resolution imaging with texture transfer network
```
@article{wu2022infrared,
  title={Infrared and visible light dual-camera super-resolution imaging with texture transfer network},
  author={Wu, Yubin and Cheng, Lianglun and Wang, Tao and Wu, Heng},
  journal={Signal Processing: Image Communication},
  volume={108},
  pages={116825},
  year={2022},
  publisher={Elsevier}
}
```

### GAN based

Generative adversarial networks
```
@article{goodfellow2020generative,
  title={Generative adversarial networks},
  author={Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  journal={Communications of the ACM},
  volume={63},
  number={11},
  pages={139--144},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```
Structure-preserving super resolution with gradient guidance
```
@inproceedings{ma2020structure,
  title={Structure-preserving super resolution with gradient guidance},
  author={Ma, Cheng and Rao, Yongming and Cheng, Yean and Chen, Ce and Lu, Jiwen and Zhou, Jie},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={7769--7778},
  year={2020}
}
```
Improved training of wasserstein gans
```
@article{gulrajani2017improved,
  title={Improved training of wasserstein gans},
  author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron C},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
Single frame infrared image super-resolution algorithm based on generative adversarial nets
```
@article{shao2018single,
  title={Single frame infrared image super-resolution algorithm based on generative adversarial nets},
  author={SHAO, Baotai and TANG, Xinyi and JIN, Lu and Li, Zheng},
  journal={Journal of Infrared and Millimeter Wave},
  volume={37},
  number={4},
  pages={427--432},
  year={2018}
}
```
Deep learning enhancement of infrared face images using generative adversarial networks
```
@article{guei2018deep,
  title={Deep learning enhancement of infrared face images using generative adversarial networks},
  author={Guei, Axel-Christian and Akhloufi, Moulay},
  journal={Applied optics},
  volume={57},
  number={18},
  pages={D98--D107},
  year={2018},
  publisher={Optical Society of America}
}
```
Single infrared remote sensing image super-resolution via supervised deep learning
```
@inproceedings{zhang2020single,
  title={Single infrared remote sensing image super-resolution via supervised deep learning},
  author={Zhang, Cong and Zhang, Haopeng and Jiang, Zhiguo},
  booktitle={Image and Signal Processing for Remote Sensing XXVI},
  volume={11533},
  pages={241--249},
  year={2020},
  organization={SPIE}
}
```
Feedback network for image super-resolution
```
@inproceedings{li2019feedback,
  title={Feedback network for image super-resolution},
  author={Li, Zhen and Yang, Jinglei and Liu, Zheng and Yang, Xiaomin and Jeon, Gwanggil and Wu, Wei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={3867--3876},
  year={2019}
}
```
A Novel Domain Transfer-Based Approach for Unsupervised Thermal Image Super-Resolution
```
@article{rivadeneira2022novel,
  title={A Novel Domain Transfer-Based Approach for Unsupervised Thermal Image Super-Resolution},
  author={Rivadeneira, Rafael E and Sappa, Angel D and Vintimilla, Boris X and Hammoud, Riad},
  journal={Sensors},
  volume={22},
  number={6},
  pages={2254},
  year={2022},
  publisher={MDPI}
}
```
Infrared image super-resolution reconstruction by using generative adversarial network with an attention mechanism
```
@article{liu2021infrared,
  title={Infrared image super-resolution reconstruction by using generative adversarial network with an attention mechanism},
  author={Liu, Qing-Ming and Jia, Rui-Sheng and Liu, Yan-Bo and Sun, Hai-Bin and Yu, Jian-Zhi and Sun, Hong-Mei},
  journal={Applied Intelligence},
  volume={51},
  number={4},
  pages={2018--2030},
  year={2021},
  publisher={Springer}
}
```
LATIS: Lambda Abstraction-based Thermal Image Super-resolution
```
@article{panda2023latis,
  title={LATIS: Lambda Abstraction-based Thermal Image Super-resolution},
  author={Panda, Gargi and Kundu, Soumitra and Bhattacharya, Saumik and Routray, Aurobinda},
  journal={arXiv preprint arXiv:2311.12046},
  year={2023}
}
```

Multimodal sensor fusion in single thermal image super-resolution
```
@inproceedings{almasri2018multimodal,
  title={Multimodal sensor fusion in single thermal image super-resolution},
  author={Almasri, Feras and Debeir, Olivier},
  booktitle={Asian Conference on Computer Vision},
  pages={418--433},
  year={2018},
  organization={Springer}
}
```
Infrared Image Super-Resolution via Heterogeneous Convolutional WGAN
```
@inproceedings{huang2021pricai,
  title={Infrared Image Super-Resolution via Heterogeneous Convolutional WGAN},
  author={Huang, Yongsong and Jiang, Zetao and Wang, Qingzhong and Jiang, Qi and Pang, Guoming},
  booktitle={Pacific Rim International Conference on Artificial Intelligence},
  pages={461--472},
  year={2021},
  organization={Springer}
}
```
Deep learning-based thermal image reconstruction and object detection
```
@article{batchuluun2020deep,
  title={Deep learning-based thermal image reconstruction and object detection},
  author={Batchuluun, Ganbayar and Kang, Jin Kyu and Nguyen, Dat Tien and Pham, Tuyen Danh and Arsalan, Muhammad and Park, Kang Ryoung},
  journal={IEEE Access},
  volume={9},
  pages={5951--5971},
  year={2020},
  publisher={IEEE}
}
```
A joint cross-modal super-resolution approach for vehicle detection in aerial imagery
```
@inproceedings{mostofa2020joint,
  title={A joint cross-modal super-resolution approach for vehicle detection in aerial imagery},
  author={Mostofa, Moktari and Ferdous, Syeda Nyma and Nasrabadi, Nasser M},
  booktitle={Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications II},
  volume={11413},
  pages={184--194},
  year={2020},
  organization={SPIE}
}
```
Thermal Image Super-resolution: A Novel Architecture and Dataset
```
@inproceedings{rivadeneira2020thermal,
  title={Thermal Image Super-resolution: A Novel Architecture and Dataset.},
  author={Rivadeneira, Rafael E and Sappa, Angel D and Vintimilla, Boris Xavier},
  booktitle={VISIGRAPP (4: VISAPP)},
  pages={111--119},
  year={2020}
}
```
Thermal image reconstruction using deep learning
```
@article{batchuluun2020thermal,
  title={Thermal image reconstruction using deep learning},
  author={Batchuluun, Ganbayar and Lee, Young Won and Nguyen, Dat Tien and Pham, Tuyen Danh and Park, Kang Ryoung},
  journal={IEEE Access},
  volume={8},
  pages={126839--126858},
  year={2020},
  publisher={IEEE}
}

```
# 4 Datasets and IQA

## Datasets

A Survey on Infrared Image and Video Sets
```
@article{danaci2022survey,
  title={A Survey on Infrared Image and Video Sets},
  author={Danaci, Kevser Irem and Akagunduz, Erdem},
  journal={arXiv preprint arXiv:2203.08581},
  year={2022}
}
```
A standard data set for performance analysis of advanced IR image processing techniques
```
@inproceedings{weiss2012standard,
  title={A standard data set for performance analysis of advanced IR image processing techniques},
  author={Wei{\ss}, A Robert and Adomeit, Uwe and Chevalier, Philippe and Landeau, St{\'e}phane and Bijl, Piet and Champagnat, Fr{\'e}d{\'e}ric and Dijk, Judith and G{\"o}hler, Benjamin and Landini, Stefano and Reynolds, Joseph P and others},
  booktitle={Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XXIII},
  volume={8355},
  pages={354--363},
  year={2012},
  organization={SPIE}
}
```

Eigen-patch iris super-resolution for iris recognition improvement
```
@inproceedings{alonso2015eigen,
  title={Eigen-patch iris super-resolution for iris recognition improvement},
  author={Alonso-Fernandez, Fernando and Farrugia, Reuben A and Bigun, Josef},
  booktitle={2015 23rd European Signal Processing Conference (EUSIPCO)},
  pages={76--80},
  year={2015},
  organization={IEEE}
}
```
RGB-IR cross input and sub-pixel upsampling network for infrared image super-resolution
```
@article{du2020rgb,
  title={RGB-IR cross input and sub-pixel upsampling network for infrared image super-resolution},
  author={Du, Juan and Zhou, Huixin and Qian, Kun and Tan, Wei and Zhang, Zhe and Gu, Lin and Yu, Yue},
  journal={Sensors},
  volume={20},
  number={1},
  pages={281},
  year={2020},
  publisher={MDPI}
}
```

Infrared and visible image fusion with convolutional neural networks
```
@article{liu2018infrared1,
  title={Infrared and visible image fusion with convolutional neural networks},
  author={Liu, Yu and Chen, Xun and Cheng, Juan and Peng, Hu and Wang, Zengfu},
  journal={International Journal of Wavelets, Multiresolution and Information Processing},
  volume={16},
  number={03},
  pages={1850018},
  year={2018},
  publisher={World Scientific}
}
```
Infrared and visual image fusion through infrared feature extraction and visual information preservation
```
@article{zhang2017infrared1,
  title={Infrared and visual image fusion through infrared feature extraction and visual information preservation},
  author={Zhang, Yu and Zhang, Lijia and Bai, Xiangzhi and Zhang, Li},
  journal={Infrared Physics \& Technology},
  volume={83},
  pages={227--237},
  year={2017},
  publisher={Elsevier}
}
```

### IQA

Image quality assessment: from error visibility to structural similarity
```
@article{wang2004image,
  title={Image quality assessment: from error visibility to structural similarity},
  author={Wang, Zhou and Bovik, Alan C and Sheikh, Hamid R and Simoncelli, Eero P},
  journal={IEEE transactions on image processing},
  volume={13},
  number={4},
  pages={600--612},
  year={2004},
  publisher={IEEE}
}
```
Why is image quality assessment so difficult?
```
@inproceedings{wang2002image,
  title={Why is image quality assessment so difficult?},
  author={Wang, Zhou and Bovik, Alan C and Lu, Ligang},
  booktitle={2002 IEEE International conference on acoustics, speech, and signal processing},
  volume={4},
  pages={IV--3313},
  year={2002},
  organization={IEEE}
}
```
A statistical evaluation of recent full reference image quality assessment algorithms
```
@article{sheikh2006statistical,
  title={A statistical evaluation of recent full reference image quality assessment algorithms},
  author={Sheikh, Hamid R and Sabir, Muhammad F and Bovik, Alan C},
  journal={IEEE Transactions on image processing},
  volume={15},
  number={11},
  pages={3440--3451},
  year={2006},
  publisher={IEEE}
}
```
Mean squared error: Love it or leave it? A new look at signal fidelity measures
```
@article{wang2009mean,
  title={Mean squared error: Love it or leave it? A new look at signal fidelity measures},
  author={Wang, Zhou and Bovik, Alan C},
  journal={IEEE signal processing magazine},
  volume={26},
  number={1},
  pages={98--117},
  year={2009},
  publisher={IEEE}
}
```
Making a “completely blind” image quality analyzer
```
@article{mittal2012making,
  title={Making a “completely blind” image quality analyzer},
  author={Mittal, Anish and Soundararajan, Rajiv and Bovik, Alan C},
  journal={IEEE Signal processing letters},
  volume={20},
  number={3},
  pages={209--212},
  year={2012},
  publisher={IEEE}
}
```
The unreasonable effectiveness of deep features as a perceptual metric
```
@inproceedings{zhang2018unreasonable,
  title={The unreasonable effectiveness of deep features as a perceptual metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={586--595},
  year={2018}
}
```
Infrared image super-resolution method for edge computing based on adaptive nonlocal means
```
@article{sun2022infrared,
  title={Infrared image super-resolution method for edge computing based on adaptive nonlocal means},
  author={Sun, Tao and Xiong, Zhengqiang and Wei, Zixian and Wang, Zhengxing},
  journal={The Journal of Supercomputing},
  volume={78},
  number={5},
  pages={6717--6738},
  year={2022},
  publisher={Springer}
}
```
No-reference image quality assessment in the spatial domain
```
@article{mittal2012no,
  title={No-reference image quality assessment in the spatial domain},
  author={Mittal, Anish and Moorthy, Anush Krishna and Bovik, Alan Conrad},
  journal={IEEE Transactions on image processing},
  volume={21},
  number={12},
  pages={4695--4708},
  year={2012},
  publisher={IEEE}
}
```

EO system design and performance optimization by image-based end-to-end modeling

```
@inproceedings{bijl2019eo,
  title={EO system design and performance optimization by image-based end-to-end modeling},
  author={Bijl, Piet and Hogervorst, MA},
  booktitle={Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XXX},
  volume={11001},
  pages={177--191},
  year={2019},
  organization={SPIE}
}
```

A holistic approach to high performance infrared system design
```
@inproceedings{driggers2018holistic,
  title={A holistic approach to high performance infrared system design},
  author={Driggers, R and Vollmerhausen, R and Short, R and Littlejohn, D and Scholten, M},
  booktitle={Infrared Technology and Applications XLIV},
  volume={10624},
  pages={58--67},
  year={2018},
  organization={SPIE}
}
```

Progress in sensor performance testing, modeling and range prediction using the TOD method: an overview
```
@article{bijl2017progress,
  title={Progress in sensor performance testing, modeling and range prediction using the TOD method: an overview},
  author={Bijl, Piet and Hogervorst, Maarten A and Toet, Alexander},
  journal={Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XXVIII},
  volume={10178},
  pages={179--197},
  year={2017},
  publisher={SPIE}
}
```
Target Acquisition performance: effects of target aspect angle, dynamic imaging and signal processing. In: Infrared Imaging Systems: Design, Analysis, Modeling, and Testing
```
@inproceedings{beintema2008target,
  title={Target acquisition performance: effects of target aspect angle, dynamic imaging, and signal processing},
  author={Beintema, Jaap A and Bijl, Piet and Hogervorst, Maarten A and Dijk, Judith},
  booktitle={Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XIX},
  volume={6941},
  pages={88--99},
  year={2008},
  organization={SPIE}
}
```

 TOD predicts target acquisition performance for staring  and scanning thermal imagers
 ```
 @inproceedings{bijl2000tod,
  title={TOD predicts target acquisition performance for staring and scanning thermal imagers},
  author={Bijl, Piet and Valeton, J Mathieu and de Jong, Arie N},
  booktitle={Infrared Imaging Systems: Design, Analysis, Modeling, and Testing XI},
  volume={4030},
  pages={96--103},
  year={2000},
  organization={SPIE}
}
```

# 5 Trends




## Diffusion model

High-resolution image synthesis with latent diffusion models
```
@inproceedings{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10684--10695},
  year={2022}
}
```
Progressive distillation for fast sampling of diffusion models
```
@article{salimans2022progressive,
  title={Progressive distillation for fast sampling of diffusion models},
  author={Salimans, Tim and Ho, Jonathan},
  journal={arXiv preprint arXiv:2202.00512},
  year={2022}
}
```
Accelerating Diffusion Models via Early Stop of the Diffusion Process
```
@article{lyu2022accelerating,
  title={Accelerating Diffusion Models via Early Stop of the Diffusion Process},
  author={Lyu, Zhaoyang and Xu, Xudong and Yang, Ceyuan and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2205.12524},
  year={2022}
}
```
A survey on generative diffusion model
```
@article{cao2022survey,
  title={A survey on generative diffusion model},
  author={Cao, Hanqun and Tan, Cheng and Gao, Zhangyang and Chen, Guangyong and Heng, Pheng-Ann and Li, Stan Z},
  journal={arXiv preprint arXiv:2209.02646},
  year={2022}
}
```

## Transformer

Attention is all you need
```
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
Transformer in transformer
```
@article{han2021transformer,
  title={Transformer in transformer},
  author={Han, Kai and Xiao, An and Wu, Enhua and Guo, Jianyuan and Xu, Chunjing and Wang, Yunhe},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={15908--15919},
  year={2021}
}
```
Swin transformer: Hierarchical vision transformer using shifted windows
```
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```
A survey on visual transformer
```
@article{han2020survey,
  title={A survey on visual transformer},
  author={Han, Kai and Wang, Yunhe and Chen, Hanting and Chen, Xinghao and Guo, Jianyuan and Liu, Zhenhua and Tang, Yehui and Xiao, An and Xu, Chunjing and Xu, Yixing and others},
  journal={arXiv preprint arXiv:2012.12556},
  volume={2},
  number={4},
  year={2020}
}
```
Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation
```
@article{wei2022contrastive,
  title={Contrastive Learning Rivals Masked Image Modeling in Fine-tuning via Feature Distillation},
  author={Wei, Yixuan and Hu, Han and Xie, Zhenda and Zhang, Zheng and Cao, Yue and Bao, Jianmin and Chen, Dong and Guo, Baining},
  journal={arXiv preprint arXiv:2205.14141},
  year={2022}
}
```
Focal Modulation Networks
```
@article{yang2022focal,
  title={Focal Modulation Networks},
  author={Yang, Jianwei and Li, Chunyuan and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2203.11926},
  year={2022}
}
```
Coca: Contrastive captioners are image-text foundation models
```
@article{yu2022coca,
  title={Coca: Contrastive captioners are image-text foundation models},
  author={Yu, Jiahui and Wang, Zirui and Vasudevan, Vijay and Yeung, Legg and Seyedhosseini, Mojtaba and Wu, Yonghui},
  journal={arXiv preprint arXiv:2205.01917},
  year={2022}
}
```
Musiq: Multi-scale image quality transformer
```
@inproceedings{ke2021musiq,
  title={Musiq: Multi-scale image quality transformer},
  author={Ke, Junjie and Wang, Qifei and Wang, Yilin and Milanfar, Peyman and Yang, Feng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5148--5157},
  year={2021}
}
```
Pyramid adversarial training improves vit performance
```
@inproceedings{herrmann2022pyramid,
  title={Pyramid adversarial training improves vit performance},
  author={Herrmann, Charles and Sargent, Kyle and Jiang, Lu and Zabih, Ramin and Chang, Huiwen and Liu, Ce and Krishnan, Dilip and Sun, Deqing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13419--13429},
  year={2022}
}
```
ViTGAN: Training GANs with Vision Transformers
```
@inproceedings{lee2021vitgan,
  title={ViTGAN: Training GANs with Vision Transformers},
  author={Lee, Kwonjoon and Chang, Huiwen and Jiang, Lu and Zhang, Han and Tu, Zhuowen and Liu, Ce},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
## Blind SR

A Closer Look at Blind Super-Resolution: Degradation Models, Baselines, and Performance Upper Bounds
```
@inproceedings{zhang2022closer,
  title={A Closer Look at Blind Super-Resolution: Degradation Models, Baselines, and Performance Upper Bounds},
  author={Zhang, Wenlong and Shi, Guangyuan and Liu, Yihao and Dong, Chao and Wu, Xiao-Ming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={527--536},
  year={2022}
}
```
From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution
```

@article{li2022face,
  title={From Face to Natural Image: Learning Real Degradation for Blind Image Super-Resolution},
  author={Li, Xiaoming and Chen, Chaofeng and Lin, Xianhui and Zuo, Wangmeng and Zhang, Lei},
  journal={arXiv preprint arXiv:2210.00752},
  year={2022}
}
```
Degradation-Guided Meta-Restoration Network for Blind Super-Resolution
```

@article{yang2022degradation,
  title={Degradation-Guided Meta-Restoration Network for Blind Super-Resolution},
  author={Yang, Fuzhi and Yang, Huan and Zeng, Yanhong and Fu, Jianlong and Lu, Hongtao},
  journal={arXiv preprint arXiv:2207.00943},
  year={2022}
}
```
Joint Learning Content and Degradation Aware Feature for Blind Super-Resolution
```
@inproceedings{zhou2022joint,
  title={Joint Learning Content and Degradation Aware Feature for Blind Super-Resolution},
  author={Zhou, Yifeng and Lin, Chuming and Luo, Donghao and Liu, Yong and Tai, Ying and Wang, Chengjie and Chen, Mingang},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={2606--2616},
  year={2022}
}
```
Learning Multiple Probabilistic Degradation Generators for Unsupervised Real World Image Super Resolution
```
@article{lee2022learning,
  title={Learning Multiple Probabilistic Degradation Generators for Unsupervised Real World Image Super Resolution},
  author={Lee, Sangyun and Ahn, Sewoong and Yoon, Kwangjin},
  journal={arXiv preprint arXiv:2201.10747},
  year={2022}
}
```
Rethinking Degradation: Radiograph Super-Resolution via AID-SRGAN
```
@article{huang2022rethinking,
  title={Rethinking Degradation: Radiograph Super-Resolution via AID-SRGAN},
  author={Huang, Yongsong and Wang, Qingzhong and Omachi, Shinichiro},
  journal={arXiv preprint arXiv:2208.03008},
  year={2022}
}
```
Toward real-world super-resolution via adaptive downsampling models
```
@article{son2021toward,
  title={Toward real-world super-resolution via adaptive downsampling models},
  author={Son, Sanghyun and Kim, Jaeha and Lai, Wei-Sheng and Yang, Ming-Hsuan and Lee, Kyoung Mu},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2021},
  publisher={IEEE}
}
```


### End
