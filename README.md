# TSP_project
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://boutainaelyaziji-tsp-project-app-cod2bj.streamlit.app/)


The objective of this project is to solve the problem of the traveling salesman. It is a
mathematical problem which consists, given a set of cities separated by given distances,
in finding the shortest path which connects all the cities.
We have implemented several methods to try to solve this problem :<br/> an exact method
represented by dynamic programming as well as another metaheuristic method (GVNS).

<img alt="TSP App"src="https://github.com/BoutainaELYAZIJI/TSP_project/blob/main/imgs/HomePage.png"/>

### Description : 
<p>
The given instruction provides a user interface for selecting an approach from the sidebar to perform a task related to the Traveling Salesman Problem (TSP). To start with, the user needs to upload an Excel or CSV file. The provided dataset named "TSP Maroc Data" is recommended to test the approach.
</p>
<p>
Next, the user is required to enter the name of the sheet in the uploaded file and the name of the first city to initiate the computation. The approach then explores the minimum distance between different cities and generates the minimum path that the salesman should take to visit all the cities and return to the starting point.
</p>
<p>
The user can also generate a visualization graph to view the minimum path and the distance between cities. Once the visualization is generated, the user can download the graph for future reference. This approach can be helpful for businesses and organizations that require optimized route planning for efficient transportation and logistics.
 
</p>

### Demo  __click on the link below__ :
 
<div align="center" >

<a style href="https://boutainaelyaziji-tsp-project-app-cod2bj.streamlit.app/" >
<img  src="https://github.com/BoutainaELYAZIJI/TSP_project/blob/main/imgs/HomePage.png" >
<p>Click here to see the demo</p>
</a>

</div>

### Install

```shell script
pip install streamlit
pip install pandas 
pip install streamlit_option_menu
pip install matplotlib
pip install numpy
pip install streamlit-lottie

```

### Run

Streamlit need to run for development mode.

```shell script
streamlit run App.py
```

