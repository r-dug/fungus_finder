'''
This is a program to reorganize the iNat dataset I have, so that I can see how well training on the data works when organized differently.
For instance, is the imagenetV3 architecture unable to perform image classification at an acceptable level of performance at one level 
    of specificity (specificity referring to taxanomic rank)? Maybe classification with less specificity will result in higher accuracy...
This hypothesis makes some sense intuitively, given the similarity across certain species...
It also stands to reason because of the plateau effect I have been noticing trying to classify the whole iNat dataset,
    which has a whopping 10,000 classes. The accuracy of models trained on this ends up plateauing at an accuracy I find completely unacceptable.

Ideally, I want to still achieve some reasonable species level classification, though. 
I think that it might make sense to focus in on smaller class sets once we have zoned in on saaaay... a genus level of specificity, for example.
Maybe if the inference is above a level of certainty for the first classification, we move on to classifying the finer level of specificity. 
I envision a tradeoff to be made, where maybe great performance is achievable through fewer steps... 

I will test this hypothesis to find out... 
    Because tensor flow uses the names of immediate child directories to the one selected as the target dataset
    as labels for classes, I am opting to select a level of specificity, then sort to that level, train the model
    and record the results of that level of specificity, then repeat that process for each level of specificity and compare.

Methodology:
1) Reorganize dataset into desired level of specificity using this script.
    
2) Train a model on resorted dataset & record results.

3) Repeat, iterating through each level of taxonomic rank.

4) Analyze results.
'''
import os
import sys

usage_string = """
Usage: 
> python3 reorg.py [integer corresponding to level of taxonomic rank to sort files into]
Taxon rank levels:
1 - Kingdom
2 - Phylum
3 - Class
4 - Order
5 - Family
6 - Genus
7 - Species
"""
taxon_rank = int(sys.argv[1])
dirlist = []
for root, dirs, files in os.walk("./DATASETS/train/",topdown=True):
    for dir in dirs:
        taxon = dir.split('_')[taxon_rank]
        if taxon not in dirlist:
            dirlist.append(taxon)
            os.system(f"mkdir {root}{taxon}")
        os.system(f"mv {root}{dir} {root}{taxon}")
        continue
print(dirlist)
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print()
        exit()