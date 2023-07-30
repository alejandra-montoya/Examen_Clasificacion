import splitfolders

inputFolder = 'Rice_Image_Dataset/Train'
outputFolder = 'Rice_Image_Dataset/Split'

splitfolders.ratio(inputFolder,outputFolder,seed=42, ratio=(0.6,0.2,0.2))