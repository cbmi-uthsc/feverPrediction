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
    <li><b>Temperature</b>: Patient’s temperature value in celsius/li>
    <li><b>saO2</b>: Patient’s saO2 value e.g.: 99, 94, 98</li>
    <li><b>heartRate</b>: Patient’s heart rate value e.g.: 102, 104, 70 </li>
    <li><b>respiration</b>: Patient’s respiration value e.g.: 25, 20, 17</li>
    <li><b>cvp</b>: Patient’s cvp value e.g.: 359, 272, 293</li>
    <li><b>systemicSystolic</b>: Patient’s systolic value e.g.: 120, 103, 106</li>
    <li><b>systemicDiastolic</b>: Patient’s diastolic value e.g.: 73, 65, 63</li>
    <li><b>systemicMean</b>: Patient’s mean pressure e.g.: 89, 75, 78</li>
</ol>


## Prototype I
You can check out first prototype <a href="https://github.com/adityauser/feverPrediction/tree/master/Prototype_I">here</a><br>.
