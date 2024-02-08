# Rosneft_hackaton

This is [Rosneft hackaton][1] which we (I and [Mikhail Vitko][2]) have participated and secured a spot in the top 10 out of 180 teams.
The hackathon took place in October 2023.

Task: Using data from geological maps and seismic survey results, develop a predictive model for unexplored areas.
---

Problem statement
---
<p align="center">
  <img src="https://github.com/MrShaller/Rosneft_hackaton/assets/62774239/134976e3-1a41-4a2d-ac57-76ea1e2d264b" alt="Image">
</p> 

In a certain region 𝑅 on the plane, several functions 𝑀𝑎𝑝𝑖 𝑥, 𝑦 , 𝑖 = 1, … , 5 are defined by their values on a regular rectangular grid. It is known that all these functions are controlled by a set of unknown interrelated functions 𝐹𝑗 𝑥, 𝑦 , from which values of only one function 𝐹1 𝑥, 𝑦 are specified at several points (not necessarily coinciding with grid nodes). It is also known that the distributions of functions 𝐹𝑗 𝑥, 𝑦 are characterized by zonality (depend on coordinates 𝑥, 𝑦 ).

It is required to find the values of the unknown function 𝐹1 𝑥, 𝑦 at all grid nodes and assess the quality of the found approximation.

<p align="center">
  <img src="https://github.com/MrShaller/Rosneft_hackaton/assets/62774239/db9cd8d3-08d8-405b-98f0-9661029de7f9" alt="Image">
</p>

Data
---
The data was presented in the following format, where (x, y) represents coordinates, and z is one of the seismic signals at a depth of 2 thousand kilometers.

The task was to determine the values of Z for the Point_dataset.txt table in unknown areas (x, y) using all 5 original maps.
<p align="center">
  <img src="https://github.com/MrShaller/Rosneft_hackaton/assets/62774239/04aaa063-032a-4b7c-8b26-8d5146289242" alt="Image">
</p>

Solution
---
After analyzing the source data, we hypothesized that each of the MAR files contains the results of geophysical studies. The set of geophysical parameters, in turn, characterizes a specific rock type, which may be of particular interest in delineating oil-bearing contours and predicting flow rates. This assumption is supported by the fact that the distribution of functions F is characterized by zonality.

After that, our team split into two: Mikhail attempted to solve the problem using clustering and classification, while I, in turn, tried to address the issue through regression. In the future, we decided to focus on regression.

The approximate solution path can be seen further:

<p align="center">
  <img src="https://github.com/MrShaller/Rosneft_hackaton/assets/62774239/9c941f4f-9f7b-4de4-b1ec-95e4b9d0244b" alt="Image">
</p>

<p align="center">
  <img src="https://github.com/MrShaller/Rosneft_hackaton/assets/62774239/b5b0d1da-918f-4e50-bb1e-afb92d7ce2ff" alt="Image">
</p>

Result
---
As a result, the RF model with regression showed 89% accuracy on the training data (as shown in the following image), and on the final test set, it achieved 91% accuracy. 
At the same time, the model with clustering showed around 80% accuracy without hyperparameter tuning.

The final file main.py includes the best approach (RF) for this task.

<p align="center">
  <img src="https://github.com/MrShaller/Rosneft_hackaton/assets/62774239/3c8bd2b0-a6a0-4d98-a9fb-7633dc1e6355" alt="Image">
</p>




[1]: https://events.rn.digital/hack/it2023vuz
[2]: https://github.com/mishantique
