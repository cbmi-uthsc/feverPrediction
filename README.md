# Fever Prediction
Fever prediction model using high-frequency real-time sensor data

<b>Problem Statement</b>: Build a Python-based application to predict fever from ICU sensor data streams.
For cases, Identify a fever episode temp >= 38, look up to 6 hours back, extract features from that window. If a patient had multiple fever episodes during their stay, treat each episode as independent if there is at least a 24h gap between them. For controls, identify patients who never had any temperature over 38 or under 34 degrees Celsius. Randomly select a 6 hour period. Build regression models to predict the onset of fever.

<h4>Table of Contents</h4>
<ol>
    <li>Introduction</li>
    <li>Modules</li>
    <li>Code Description</li>
    <li>GSoC Experience</li>
    <li>Conclusion</li>
    <li>Team</li>
    <li>License</li>
</ol>

## Introduction

### Background
Fever can provide valuable information for diagnosis and prognosis of various diseases such as pneumonia, dengue, sepsis, etc., therefore, predicting fever early can help in the effectiveness of treatment options and expediting the treatment process. The aim of this project is to develop novel algorithms that can accurately predict fever onset in critically ill patients by applying machine learning technique on continuous physiological data. We have maded a model which can predict the occurence of fever, hours before it actaully occurs. This will provide doctors to take contingency actions early, and will decrease mortality rates significantly.

### Dataset
We hace used vitialPeriodic dataset which is provided by the eICU Collaborative Research Database. It contains continuous physiological data collected every 5-minute from a cohort of over200,000 critically ill patients admitted to an Intensive Care Unit (ICU) over a 2-year period.
<h4>Physiological Variabels</h4>
<ol>
    <li> <b>Temperature</b> : Patient’s temperature value in celsius </li>
    <li> <b>saO2</b> : Patient’s saO2 value e.g.: 99, 94, 98 </li>
    <li> <b>heartRate</b> : Patient’s heart rate value e.g.: 102, 104, 70 </li>
    <li> <b>respiration</b> : Patient’s respiration value e.g.: 25, 20, 17</li>
    <li> <b>cvp</b> : Patient’s cvp value e.g.: 359, 272, 293</li>
    <li> <b>systemicSystolic</b> : Patient’s systolic value e.g.: 120, 103, 106</li>
    <li> <b>systemicDiastolic</b> : Patient’s diastolic value e.g.: 73, 65, 63</li>
    <li> <b>systemicMean</b> : Patient’s mean pressure e.g.: 89, 75, 78</li>
</ol>

## Modules

### Feature Extraction
For the feature extraction process, we need to introduce the concept of time windows and time before true onset. Preprocessing is done is such a way that the time window, i.e the amount of data in a time period required to train the model is kept constant at 10 hours. So, we always train the model using 10hrs worth of data. Time before true onset means how early do we want to predict sepsis. This parameter has been varied in steps of 2 hours to get a better understanding of how your accuracy drops off as the time difference increases. For this experiment, we have used time priors of 2, 4, 6 and 8 hours. Even the time window has sub window of 0-2 hours, 0-4 hours, 0-6 hours, 0-8 hours and 0-10 hours, the sub windows were created so that our model could get temporal idea also.
<br>
Then we have preprocessed the entire dataframe according to each of these time differences. So we have processed data for 2 hours before sepsis with 6 hours of training data, 4 hours before with 6 hours of training data and so on so forth. We have seven physiological variables data streams for 5 diffenet sub window. We then extracted 7 statistical features from each of the original 7*5 data streams. <br>
They are:
<ul>
<li>Standard Deviation</li>
<li>Kurtosis</li>
<li>Skewness</li>
<li>Mean </li>
<li>Minimum</li>
<li>Maximum</li>
<li>RMS_Difference</li>
</ul>
Therefore the net features extracted are 49*5.

### Model Development

We have tested our model on differnt models, some of them are Temporal Convolutional Networks, Logistic Regression, Random Forest and Xgboost. The data is first partitioned into the train (80%) and test (20%) datasets and then trained on the models mentioned above. Metrics like Score, F1 score and AUROC were calculated. We have got best result from Temporal Convolutional Networks.

## Team
<table align="center">
  <tbody>
    <tr>
        <td align="center" valign="top">
			<img height="150" src="https://github.com/adityauser.png?s=150">
			<br>
			<a href="https://github.com/adityauser">Aditya Singh</a>
			<br>
			<a href="mailto:adityauser225@gmail.com">adityauser225@gmail.com</a>
			<br>
			<p>Author</p>
		</td>
		<td align="center" valign="top">
			<img height="150" src="https://github.com/akram-mohammed.png?s=150">
			<br>
			<a href="https://github.com/akram-mohammed">Dr. Akram Mohammed</a>
			<br>
			<a href="mailto:akrammohd@gmail.com">akrammohd@gmail.com</a>
			<br>
			<p>Mentor, Maintainer</p>
		</td>
	 	<td align="center" valign="top">
			<img width="150" height="150" src="https://github.com/rkamaleswaran.png?s=150">
			<br>
			<a href="https://github.com/rkamaleswaran">Dr. Rishikesan Kamaleswaran</a>
			<br>
			<a href="mailto:rkamales@uthsc.edu">rkamales@uthsc.edu</a>
			<br>
			<p>Mentor</p>
		</td>
     </tr>
  </tbody>
</table>

## Prototype I
You can check out first prototype <a href="https://github.com/adityauser/feverPrediction/tree/master/Prototype_I">here</a><br>.
