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
Download the trained [model](https://drive.google.com/file/d/1j4CSF13Uoajcfh1h-yBuvVXo_-rub05o/view?usp=sharing) and unzip the files into the _model_ folder.

### Install python dependencies
Python 3.7 is required

> pip install -r requirements.txt

### Download pretrained BERTweet model
> wget https://public.vinai.io/BERTweet_base_fairseq.tar.gz

> tar -xzvf BERTweet_base_fairseq.tar.gz

In the first run, the pretrained BERT models will be automaticlly downloaded and cached on the local drive.


### Test the code
A snippet of example code is as below.

```python
from main import GazPNE2
gazpne2=GazPNE2() # This will take around 1 minute to load models
tweets = ["Associates at the Kuykendahl Rd & Louetta Rd. store in Spring, TX gave our customers a reason to smile",\
"Rockport TX any photos of damage down Corpus Christi Street and Hwy 35 area? #houstonflood"]
# It is faster to input multiple tweets at once than one single tweet mutiple times. 
locations = gazpne2.extract_location(tweets)
print(locations)
'''This will output:
{0: [{'LOC': 'Kuykendahl Rd', 'offset': (18, 30)}, {'LOC': 'Louetta Rd', 'offset': (34, 43)},
{'LOC': 'Spring', 'offset': (55, 60)}, {'LOC': 'TX', 'offset': (63, 64)}], 
1: [{'LOC': 'Corpus Christi Street', 'offset': (38, 58)}, {'LOC': 'Hwy 35', 'offset': (64, 69)},
{'LOC': 'Rockport', 'offset': (0, 7)}, {'LOC': 'TX', 'offset': (9, 10)}, {'LOC': 'houston', 'offset': (78, 84)}]}
'''
```

To extract locations from txt file, execute the following command. In the txt file, each line corresponds to a tweet message.

> python -u main.py --input=0 --input_file=data/test.txt


To test our manually annotated datasets (3000 tweets), execute the following command.

> python -u main.py --input=2

To test public datasets (19), execute the following command. You will get the result of partial datasets since some are not publicly available.

> python -u main.py --input=4
> 
datasets [a,b,c]  can be obtained from https://rebrand.ly/LocationsDataset.

datasets [e,f] can be obtained from https://revealproject.eu/geoparse-benchmark-open-dataset/.

datasets [g,h] can be obtained by contacting the [author](https://www.researchgate.net/publication/342550989_Knowledge-based_rules_for_the_extraction_of_complex_fine-grained_locative_references_from_tweets) of the data.

## Contact
If you have any questions, feel free to contact Xuke Hu via xuke.hu@dlr.de
