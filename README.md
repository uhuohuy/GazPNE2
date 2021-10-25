# GazPNE2
## Introduction

We present  a robust and general place name extraction method from tweet texts, named GazPNE2. It fuses deep learning, global gazetteers (i.e., OpenStreetMap and GeoNames), and pretrained transformer models (i.e., BERT and BERTweet), requiring no manually annotated data. It can extract place names at both coarse (e.g., country and city) and fine-grained (e.g., street and creek) levels and place names with abbreviations (e.g., ‘tx’ for ‘Texas’ and ‘studemont rd’ for ‘studemont road’). 

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
Download the trained [model](https://drive.google.com/file/d/1X1njb1UAcb6TSRijxbqH8cUo6rnUZS_M/view?usp=sharing) and unzip the files into the _model_ folder.

Download the prior probability dictionary from [here](https://drive.google.com/file/d/1jQ29fWNQP48nUUoU_Cwx67MffUjDTf1c/view?usp=sharing) and put it in the _data_ folder.

Note that, in the first run, the BERTWeeet and BERT models will be automaticlly downloaded and cached on your local drive.

### Install python dependencies

> pip install -r requirements.txt

### Run the code


> python -u main.py --input=4 --input_file=data/test.txt

To execute below command in case of a jave error.  

> spack load openjdk@11.0.8_10

Test your own data: Set <*input*> to 0 and set <*input_file*> to the path of your data. It is a .txt file with each line corresponding to a tweet message.

Test our manually annotated data (3000 tweets): Set <*input*> to 2.

Test public datasets: Set <*input*> to 4, then you will get the result of partial datasets since some are not publicly available and should be requested from the authors of the data.
datasets [a,b,c]  can be obtained from https://rebrand.ly/LocationsDataset.
datasets [e,f] can be obtained from https://revealproject.eu/geoparse-benchmark-open-dataset/.
datasets [g,h] can be obtained by contacting the [author](https://www.researchgate.net/publication/342550989_Knowledge-based_rules_for_the_extraction_of_complex_fine-grained_locative_references_from_tweets) of the data.

