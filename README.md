# Mountain Fire And Smoke Detector

Discription:
-
A simple application to help people detect the mountain fire in a large scope


Problem solution:
-
1. Based on the specific RGB formula to detect the fire.[1] </br>
2. Consider some of mountain fires are happened with the smoke first because the wet environment, so detect the smoke without specific RGB formular.</br>
  a) Using LBP or HARR training the data source.[2] </br>
  b) Tracking the smoke moving paths. </br>
  
Working processes:
-
1. Spliting the image to 3 channels image and get the R G B value. [code](https://github.com/Jayupp/Mountain_Fire_And_Smoke_Detector/blob/master/ReadMe%20source/Check_RGB_COLOR.PNG)</br> 
2. Build the positive smoke csv file and negative csv file for LBP or HARR training. [positive file code](https://github.com/Jayupp/Mountain_Fire_And_Smoke_Detector/blob/master/BuildCSV/build_positive_code.txt) [negative file code](https://github.com/Jayupp/Mountain_Fire_And_Smoke_Detector/blob/master/BuildCSV/build_negative_code.txt)<br>
After [training](https://github.com/Jayupp/Mountain_Fire_And_Smoke_Detector/blob/master/ReadMe%20source/createSamples.PNG), the test [result](https://github.com/Jayupp/Mountain_Fire_And_Smoke_Detector/blob/master/ReadMe%20source/cascade.xml) didn't show what I expect. <br>
May have these reasons:<br>
  a) The positive data source and negative source very samiliar. For emample the cloud images produce the small differences with smoke. <br>
  b) The CPU i5 is not that fast for training, some parameter limit the data accuracy.<br>






Tools
-
Visual Studio<br>
Opencv packages<br>

Developer
-
Jay Zheng<br>
All rights reversed by Jie Zheng™ 2018
