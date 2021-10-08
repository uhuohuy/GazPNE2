# GazPNE2
## Introduction

We present  a robust and general place name extraction method from tweet texts, named GazPNE2. It fuses deep learning, global gazetteers (i.e., OpenStreetMap and GeoNames), and pretrained transformer models (i.e., BERT and BERTweet), requiring no manually annotated data. It can extract place names at both coarse (e.g., country and city) and fine-grained (e.g., street and creek) levels and place names with abbreviations (e.g., ‘tx’ for ‘Texas’ and ‘studemont rd’ for ‘studemont road’). To fully evaluate GazPNE2 and competing approaches, we use 19 public tweet datasets as test data, containing in total 38,802 tweets and 22,197 place names across the world. 

## Test Data
The data we used to evaluate our approach is as follows:
<p align="center">
<a href="url">
 <img src="figure/data.png" width="700" height="470" ></a>
</p>

## Result
<p align="center">
<a href="url">
 <img src="figure/overall_result.png" ></a>
</p>

## Use the code
### Prepare model data
Download the trained [model](https://drive.google.com/file/d/1E3OenE9tKC8GiuqLYReUaljkOasxbPbL/view?usp=sharing) and unzip the data into the _model_ folder.

Note that, in the first run, the BERTWeeet and BERT models will be automaticlly downloaded and cached on your local drive.


### Run the code

> spack load openjdk@11.0.8_10

In case of a jave error, to run the above command. 


> python -u main.py --input=4 --input_file=data/2.txt --special_con_t=0.35  --abb_ent_thres=0.3 --context_thres=0.3 --weight=1 --special_ent_t=0.3 --merge_thres=0.5 
 
Test your own data: Set <*input*> to 0 and set <*input_file*> to the path of your data. It is a .txt file with each line corresponding to a tweet message.

Test our annotated data: Set <*input*> to 4, then you will get the result of partial datasets since some are not publicly available and should be requested from the authors of the data.
datasets [a,b,c]  can be obtained from https://rebrand.ly/LocationsDataset.
datasets [e,f] can be obtained from https://revealproject.eu/geoparse-benchmark-open-dataset/.
datasets [g,h] can be obtained by contacting the [author](https://www.researchgate.net/publication/342550989_Knowledge-based_rules_for_the_extraction_of_complex_fine-grained_locative_references_from_tweets) of the data.

Test our unannotated data: Set <*input*> to 6,7,8,9,10,11,12,13, or 14. This is because the datasets are too large and they are thus divided into multiple groups.

[6] correspond to CrisisNLP

[7-9] correspond to HumAID dataset

[10-14] correspond to COVID-19 dataset

The other parameters are the probability thresholds used in our approach. You can just keep the default value.
