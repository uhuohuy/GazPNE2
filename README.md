# GazPNE2
## Introduction
We present a robust and general place name extractor for English tweets, GazPNE2,  which fuses deep learning, global gazetteers (e.g., OpenStreetMap), and pretrained transformer models, requiring no manually annotated data. GazPNE2 can extract place names at both coarse (e.g., country and city) and fine-grained (e.g., street and creek) levels and place names with abbreviations (e.g., `tx' for `Texas' and `studemont rd' for `studemont road'). We compare GazPNE2 with 9 competing approaches on 11 public tweet data sets, containing 21,393 tweets and 16,790 place names across the world.
## Use the code
### Prepare model data
Download the trained [model](https://drive.google.com/file/d/1j4CSF13Uoajcfh1h-yBuvVXo_-rub05o/view?usp=sharing) and unzip the data into the _model_ folder.

Note that, in the first run, the BERTWeeet and BERT models will be automaticlly downloaded and cached in the local drive.

### Prepare test data
To test your own data, 
We have provided our test data as much as we can. The other data should be requested from the authors of the data. 


### Run the code

> spack load openjdk@11.0.8_10
> python -u main.py --input=4 --input_file=data/2.txt --special_con_t=0.35  --abb_ent_thres=0.3 --context_thres=0.3 --weight=1 --special_ent_t=0.3 --merge_thres=0.5 
> 
Test your own data: Set <*input*> to 0 and set <*input_file*> to the path of your data. It is a .txt file with each line corresponding to a tweet message.

Test our annotated data: Set <*input*> to 4, then you will get the result of partial datasets since some are not publicly available and should be requested from the authors of the data.

Test our unannotated data: Set <*input*> to 6,7,8,9,10,11,12,13,and 14. This is because the datasets are too lareg and they are divided into multiple groups.

[6] correspons to CrisisNLP

[7-9] correspons to HumAID dataset

[10-14] correspons to COVID-19 dataset

The other parameters are the probability threshold used in our approach. You can just keep the default value.


## Result
<p align="center">
<a href="url">
 <img src="figure/result.png" ></a>
</p>
