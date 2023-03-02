import streamlit as st
import pandas as pd
from typing import Optional
from streamlit_option_menu import option_menu
import random as rd
import time
from functools import lru_cache
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def f(x):
    total_cost = 0
    for c in range(1, len(x)):
        total_cost += instance[x[c - 1]][x[c]]
    return total_cost


# Neighborhood structure

def NS_swapping(x, lb, ub):  # lb>0
    bound = len(x)
    xc = None
    if (lb < bound and ub < bound):
        xc = x.copy()
        xc[lb], xc[ub] = xc[ub], xc[lb]
    return xc


def NS_insertion_before(x, lb, ub):  # lb>0
    bound = len(x)
    xc = None
    if (lb < bound and ub < bound):
        xc = x.copy()
        xc.insert(lb, x[ub])
        xc.pop(ub + 1)
    return xc


def NS_two_opt(x, lb, ub):
    bound = len(x)
    x1 = []
    if lb < ub and (0 < lb < bound - 1 and 0 < ub < bound - 1):
        x1 = x[:lb]
        x1.extend(reversed(x[lb:ub + 1]))
        x1.extend(x[ub + 1:])
    return x1


def initialization(first_city):
    bound = len(instance)
    x = [first_city]
    sol = rd.sample(range(0, bound), bound)
    for i in range(0, len(sol) - 1):
        if sol[i] == first_city:
            sol.pop(i)
    x = x + sol + x

    return x


# Shaking
def neighbrehood(x, k):
    bound = len(x)
    N = []
    if (k == 3):
        for i in range(1, bound - 2):
            for j in range(i + 1, bound - 1):
                N.append(NS_swapping(x, i, j))
    elif (k == 2):
        for i in range(1, bound - 2):
            for j in range(i + 1, bound - 1):
                N.append(NS_insertion_before(x, i, j))
    elif (k == 1):
        for i in range(1, bound - 2):
            for j in range(i + 1, bound - 1):
                N.append(NS_two_opt(x, i, j))
    return N


global voisins


def shake(x, k):
    N = neighbrehood(x, k)
    xp = rd.choice(N)
    return xp


# Changing neighberhood
def change_neighborhood(x, xp, k):
    if f(xp) < f(x):
        x = xp
    else:
        k += 1
    return x, k


# Improuve intiale solution
k_max = 3


def RVNS(x, k_max, t=5):
    start_time = time.time()
    while time.time() - start_time < t * 60:
        k = 1
        while k <= k_max:
            xp = shake(x, k)
            x, k = change_neighborhood(x, xp, k)
    return x

    # Local serach VND :


# first improvement :
def first_improvement(x, l):
    N = neighbrehood(x, l)
    for i in range(0, len(N)):
        if f(N[i]) < f(x):
            x = N[i]
            break
    return x


l_max = 2


def VND(x, l_max):
    l = 1
    while l <= l_max:
        xp = shake(x, l)
        xp = first_improvement(x, l)
        x, l = change_neighborhood(x, xp, l)
    return x


# GVNS :
def GVNS(x, t=5, k_max=3, l_max=2):
    start_time = time.time()
    x = RVNS(x, k_max, 0.2)
    while time.time() - start_time < t * 60:
        k = 1
        while k <= k_max:
            x1 = shake(x, k)
            x2 = VND(x1, l_max)
            x, k = change_neighborhood(x, x2, k)
    return x, f(x)


# function for both solutions
def randomCoordsFromMatrix(matrix) -> np.ndarray:
    lenght = len(matrix)
    coords = np.ndarray(shape=(lenght, 2), dtype=int)
    for i in range(lenght):
        for j in range(2):
            coords[i, j] = rd.randint(10, 50)
    return coords


# Dynamic Function
def symmetrize(matrix):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.
    """
    return matrix + matrix.T - np.diag(matrix.diagonal())


def solve_tsp_dynamic(
        distance_matrix: np.ndarray,
        maxsize: Optional[int] = None,
) -> Tuple[List, float]:
    N = frozenset(range(1, distance_matrix.shape[0]))
    memo: Dict[Tuple, int] = {}

    @lru_cache(maxsize=maxsize)
    def dist(ni: int, N: frozenset) -> float:
        if not N:
            return distance_matrix[ni, 0]

        costs = [
            (nj, distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
            for nj in N
        ]
        nmin, min_cost = min(costs, key=lambda x: x[1])
        memo[(ni, N)] = nmin
        return min_cost

    best_distance = dist(0, N)

    ni = 0
    solution = [1]
    while N:
        ni = memo[(ni, N)]
        solution.append(ni + 1)
        N = N.difference({ni})
    solution.append(1)

    return solution, best_distance


# inverser 2 lignes
def inversion(M, i, j):
    T = M[i, :].copy()
    M[i, :] = M[j, :]
    M[j, :] = T
    return M


def graphh(matrix, minDistance: int, permutation):
    coords = randomCoordsFromMatrix(matrix)
    markers = []
    lenght = len(matrix)
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in np.linspace(0, 1, lenght)]
    labels = [str(n + 1) for n in range(lenght)]
    chem = str(permutation[0])
    for i in range(1, len(permutation)):
        chem += ',' + str(permutation[i])

    f = plt.figure(figsize=(lenght + 4, lenght + 4))
    ax = f.add_subplot(111)
    plt.scatter(coords[:, 0], coords[:, 1], marker='o', c=colors, s=50, edgecolor='none')

    for i in range(0, lenght):
        markers.append(Line2D([0], [0], linestyle='None', marker="o", markersize=12, markeredgecolor="none",
                              markerfacecolor=colors[i]))

    lgd = plt.legend(markers, labels, numpoints=1, loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

    # Plot descriptive text
    textstr = "number of cities : %d\n First city : %d\nDistance Min: %d \n cities: %s" % (
        lenght, permutation[0], minDistance, chem)
    props = dict(boxstyle="round,pad=0.3,rounding_size=0.2", facecolor='#b3ffb3', alpha=0.5)
    plt.text(0.05, 0.98, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # Trace the target
    start_node = permutation[0]

    for i in range(0, lenght):
        start_pos = coords[start_node - 1]
        next_node = permutation[i + 1]
        end_pos = coords[next_node - 1]
        ax.annotate("",
                    xy=start_pos, xycoords='data',
                    xytext=end_pos, textcoords='data',
                    arrowprops=dict(arrowstyle="<-",
                                    connectionstyle="arc3"))

        start_node = next_node
    plt.tight_layout()
    plt.axis('equal')
    return plt


def Calculatefunction(matrix, option):
    Path, distance = solve_tsp_dynamic(matrix)
    for index, item in enumerate(Path):
        if item == option:
            Path[index] = 1
    Path[0] = Path[len(Path) - 1] = option
    col1, col2 = st.columns(2)
    col1.metric("Min Distance", f"{distance}")
    col2.metric("Min Path ", f"{Path}")
    st.header("Path Visualisation ")
    plt = graphh(matrix, int(distance), Path)
    st.pyplot(plt)
    plt.savefig("plot.png")
    with open("plot.png", "rb") as file:
        btn = st.download_button(
            label="Download The Graph",
            data=file,
            file_name="MinPath_Dynamic.png",
            mime="image/png"
        )


st.set_page_config(
    page_title="Travelling Sales Man",
    page_icon=":hello:",
    layout="wide",
)
with st.sidebar:
    selected = option_menu("TSP Solver App", ["About", "DP Approach", "GVNS Approach", "Contact"],
                           icons=['house', 'cloud-upload', "gear", 'person lines fill'], menu_icon="cast",
                           default_index=0,
                           styles={
                               "container": {"padding": "5!important", "background-color": "#fafafa"},
                               "icon": {"color": "orange", "font-size": "30px"},
                               "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px",
                                            "--hover-color": "#eee"},
                               "nav-link-selected": {"background-color": "#026CCF"},
                           }
                           )

if selected == "About":
    st.markdown("<h1 style='text-align: center;'>Welcome to Travelling Sales Man Solver App ! 👋</h1>",
                unsafe_allow_html=True)

    # st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        The traveling salesman problem (TSP) is a well-known problem in theoretical computer science and operations research. The standard version of the TSP is a hard problem and belongs to the NP-Hard class. 
        In this project, We build an application to implement the TSP by the dynamic approach and the GVNS approach.
        **👈 Select an approach from the sidebar** to try it !
        ### First upload your Excel or CSV File
        - Check out [https://github.com/BoutainaELYAZIJI/TSP_project/blob/main/tsp-maroc.xlsx]. It can be useful to test !
        ### Enter your Sheet's Name
        ### Enter your First City 
        - Explore the minimum distance 
        - Explore the minimum path
        - Generate a visualization graph 
        - Download the graph
        ### Code and Documentation 
        - Check out [https://github.com/BoutainaELYAZIJI/TSP_project]
        - Report [https://github.com/BoutainaELYAZIJI/TSP_project/blob/main/TSP.pdf]
        
        
        👈 Make sure your PC or phone is in light mode.

       
    """
    )
elif selected == "DP Approach":

    st.markdown("<h1 style='text-align: center; '>Solve TSP using Dynamic Programming 👍</h1>", unsafe_allow_html=True)
    st.markdown("""
         - **Dynamic programming** is defined as a computer programming technique where an algorithmic problem is first broken down into sub-problems, the results are saved, and then the sub-problems are optimized to find the overall solution — which usually has to do with finding the maximum and minimum range of the algorithmic query😁
    """)

    uploaded_file = st.file_uploader("Choose a CSV  file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        instance = st.text_input(
            "Enter the name of your sheet  👇",
        )
        try:
            df = pd.read_excel(uploaded_file, sheet_name=instance, index_col=False)
        except ValueError:
            st.warning('Sheets Not found or Not compatible')
        else:
            symmetrie = st.radio(
                "Do you want to make your matrice symetric ?",
                ('Yes', 'No'))
            if symmetrie == 'Yes':
                df = df.fillna(0)
                matrix = df.values

                matrix = symmetrize(matrix)
                st.dataframe(matrix)
            else:
                matrix = df.values
                st.dataframe(matrix)

            real_matrix = matrix
            option = st.text_input(
                "Enter your first city  👇",
            )
            if st.button('Calculate !'):
                try:
                    option = int(option)
                    new_matrix = inversion(real_matrix, 0, option)
                except Exception:
                    st.error('Please enter a valid number')
                else:
                    Calculatefunction(new_matrix, option)

                # print(Path, distance)

                # graphh(matrix, distance, Path)

elif selected == "GVNS Approach":
    st.markdown("<h1 style='text-align: center; '>Solve TSP using Meta-heuristic:GVNS 🔑 </h1>", unsafe_allow_html=True)
    st.markdown("""
         - GVNS, which stands for General Variable Neighborhood Search, is an established and commonly used metaheuristic for the 
         expeditious solution of optimization problems that belong to the NP-hard class 😎
    """)
    uploaded_file = st.file_uploader("Choose a CSV  file", type=['csv', 'xlsx'])

    if uploaded_file is not None:

        try:

            instance = st.text_input(
                "Enter the name of your sheet  👇")
            df = pd.read_excel(uploaded_file, sheet_name=instance, index_col=False)
        except ValueError:
            st.warning('Sheets Not found or not compatible')
        else:
            symmetrie = st.radio(
                "Do you want to make your matrice symetric ?",
                ('Yes', 'No'))
            if symmetrie == 'Yes':
                df = df.fillna(0)
                matrix = df.values

                matrix = symmetrize(matrix)
                st.dataframe(matrix)
            else:
                matrix = df.values
                st.dataframe(matrix)

        # city_option = int(st.text_input('Type 1 ,2 or 3'))
            real_matrix = matrix

            try:
                option = int(st.text_input(
                    "Enter your first city  👇",
                ))
            except ValueError or IndexError:
                st.warning('Please enter a Number')
            else:

                x = initialization((option - 1))
                temps = int(st.selectbox('Time ?', (1, 4, 8)))
                if st.button('Calculate !'):
                    i = 0
                    solution, dist_min = GVNS(x, temps)
                    for i in range(0, len(solution)):
                        solution[i] += 1
                    col1, col2 = st.columns(2)
                    col1.metric("Min Distance", f"{dist_min}")
                    col2.metric("Min Path ", f"{solution}")
                    st.header("Path Visualisation ")
                    plt = graphh(instance, int(dist_min), solution)
                    st.pyplot(plt)
                    plt.savefig("plot.png")
                    with open("plot.png", "rb") as file:
                        btn = st.download_button(
                            label="Download The Graph",
                            data=file,
                            file_name="MinPath_GVNS.png",
                            mime="image/png"
                        )

            # Start

else:
    st.header(":mailbox: Connect With Us !")
    st.markdown("""
        We would love to respond to your questions. Feel Free to get in touch with us.
    """)
    contact_form = """
            <form action="https://formsubmit.co/boutainait@gmail.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <label for="name">Full name</label><br>
            <input type="text" name="name" placeholder="Your name.."required><br>
            <label for="email">Email</label><br>
            <input type="email" name="email" placeholder="Your email.." required><br>
            <label for="msg">Subject</label><br>
               <textarea name="msg" placeholder="Write something.." style="height:200px">
             </textarea> <br>
             <button type="submit">Send</button>
        </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)


    # Use CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("style/style.css")
hide_streamlit_style = """
        <style>
         #MainMenu {visibility: hidden;}
         .css-18e3th9 {
                    padding-top: 2rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
"""

# Remove whitespace from the top of the page and sidebar

st.markdown(hide_streamlit_style, unsafe_allow_html=True)


padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
