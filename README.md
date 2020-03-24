# choking-detection
Instruction for all training videos, its names need to follow the type "videoname_number_". The number in the video's name represent that from which second, the choking can be detected. The file data_loader will automatically generate the label based on the name. 
For example, choking1_5_ means starting from 5s, the video changes from none-choking into choking. After video is cutted into 2 second clips. the clips from (0s,2s), (1s,3s), (2s,4s), (3s,5s) will be labelled none-choking, while the rest of clips will be labelled choking. 
