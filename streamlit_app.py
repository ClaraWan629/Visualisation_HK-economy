import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pygal

uber_path = "D:/Sample_Streamlit/test_visualization/datasets/uber-raw-data-apr14.csv"
ny_path = "D:/Sample_Streamlit/test_visualization/datasets/ny-trips-data.csv"
netflix = "D:/Sample_Streamlit/test_visualization/datasets/netflix_titles.csv"
gdp_path = "D:/Sample_Streamlit/test_visualization/datasets/GDP_clean.csv"
export_path = "D:/Sample_Streamlit/test_visualization/datasets/testmap.csv"

def myDecorator(function):
    def modified_function(df):
        time_ = time.time()
        res = function(df)
        time_ = time.time()-time_
       # with open(f"{function.__name__}_exec_time.txt","w") as f:
       #     f.write(f"{time_}")
        return res
    return modified_function


@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df

@myDecorator
@st.cache
def df1_data_transformation(df_):
    df = df_.copy()
    df["Date/Time"] = df["Date/Time"].map(pd.to_datetime)

    def get_dom(dt):
        return dt.day
    def get_weekday(dt):
        return dt.weekday()
    def get_hours(dt):
        return dt.hour

    df["weekday"] = df["Date/Time"].map(get_weekday)
    df["dom"] = df["Date/Time"].map(get_dom)
    df["hours"] = df["Date/Time"].map(get_hours)

    return df

@myDecorator
@st.cache
def df2_data_transformation(df_):
    df = df_.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    def get_hours(dt):
        return dt.hour

    df["hours_pickup"] = df["tpep_pickup_datetime"].map(get_hours)
    df["hours_dropoff"] = df["tpep_dropoff_datetime"].map(get_hours)

    return df

@st.cache(allow_output_mutation=True)
def frequency_by_dom(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title("Frequency by DoM - Uber - April 2014")
    ax.set_xlabel("Date of the month")
    ax.set_ylabel("Frequency")
    ax = plt.hist(x=df.dom, bins=30, rwidth=0.8, range=(0.5,30.5))
    return fig

@st.cache
def map_data(df):
    df_ = df[["Lat","Lon"]]
    df_.columns=["lat","lon"]
    return df_

@st.cache(allow_output_mutation=True)
def data_by(by,df):
    def count_rows(rows):
        return len(rows)
    
    if by == "dom":
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].set_ylim(40.72,40.75)
        ax[0].bar(x=sorted(set(df["dom"])),height=df[["dom","Lat"]].groupby("dom").mean().values.flatten())
        ax[0].set_title("Average latitude by day of the month")

        ax[1].set_ylim(-73.96,-73.98)
        ax[1].bar(x=sorted(set(df["dom"])),height=df[["dom","Lon"]].groupby("dom").mean().values.flatten(), color="orange")
        ax[1].set_title("Average longitude by day of the month")
        return fig
    
    elif by == "hours":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.hours, bins=24, range=(0.5,24))
        return fig
    
    elif by == "dow":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.weekday, bins=7, range=(-5,6.5))
        return fig
    
    elif by == "dow_xticks":
        fig, ax= plt.subplots(figsize=(10,6))
        ax.set_xticklabels('Mon Tue Wed Thu Fri Sat Sun'.split())
        ax.set_xticks(np.arange(7))
        ax = plt.hist(x=df.weekday, bins=7, range=(0,6))
        return fig
    
    else:
        pass

@st.cache
def group_by_wd(df):    
    def count_rows(rows):
        return len(rows)
    grp_df = df.groupby(["weekday","hours"]).apply(count_rows).unstack()
    return grp_df

@st.cache(allow_output_mutation=True)
def grp_heatmap(grp_df):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.heatmap(grp_df)
    return fig

@st.cache(allow_output_mutation=True)
def lat_lon_hist(df,fusion=False):
    lat_range = (40.5,41)
    lon_range = (-74.2,-73.6)

    if fusion:
        fig, ax = plt.subplots()
        ax1 = ax.twiny()
        ax.hist(df.Lon, range=lon_range, color="yellow")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Frequency")

        ax1.hist(df.Lat, range=lat_range)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Frequency")
        return fig
    
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))


        ax[0].hist(df.Lat, range=lat_range, color="red")
        ax[0].set_xlabel("Latitude")
        ax[0].set_ylabel("Frequence")

        ax[1].hist(df.Lon, range=lon_range, color="green")
        ax[1].set_xlabel("Longitude")
        ax[1].set_ylabel("Frequence")
        return fig

@st.cache(allow_output_mutation=True)
def display_points(data, color=None):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.scatterplot(data=data) if color == None else sns.scatterplot(data=data, color=color)
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(10,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").sum().values.flatten(), color="red")
    ax[0,0].set_title("Total Number of passengers per pickup hour")

    ax[0,1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").mean().values.flatten(), color="yellow")
    ax[0,1].set_title("Average Number of passengers per pickup hour")

    ax[1,0].bar(x=sorted(set(df["hours_pickup"])), height=df["hours_pickup"].value_counts().sort_index().values.flatten(), color="green")
    ax[1,0].set_title("Total number of passages per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_dropoff_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(12,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").sum().values.flatten())
    ax[0,0].set_title("Total Number of passengers per dropoff hour")

    ax[0,1].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").mean().values.flatten(), color="black")
    ax[0,1].set_title("Average Number of passengers per dropoff hour")

    ax[1,0].bar(x=sorted(set(df["hours_dropoff"])), height=df["hours_dropoff"].value_counts().sort_index().values.flatten(), color="orange")
    ax[1,0].set_title("Total number of passages per dropoff hour")
    return fig

@st.cache(allow_output_mutation=True)
def amount_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(12,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").sum().values.flatten(), color="grey")
    ax[0].set_title("Total trip distance per pickup hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").mean().values.flatten())
    ax[1].set_title("Average trip distance per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def distance_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(10,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").sum().values.flatten(), color="lime")
    ax[0].set_title("Total amount per hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").mean().values.flatten(), color="pink")
    ax[1].set_title("Average amount per hour")
    return fig

@st.cache(allow_output_mutation=True)
def corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.heatmap(df.corr())
    return fig

@st.cache(allow_output_mutation=True)
def lineplot_trend(df):
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x="Year", y="GDP at current market prices", color='green')
    #plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['2000', '2002', '2004', '2006', '2008', '2010', '2012', '2014', '2016', '2018'])
    fig = plt.gcf()
    fig_size = fig.get_size_inches()  # Get current size
    sizefactor = 0.7  # Set a zoom factor
    # Modify the current size by the factor
    fig.set_size_inches(sizefactor * fig_size)
    fig.savefig('lineplot_GDP.png')
    return fig

@st.cache(allow_output_mutation=True)
def boxplot_dist(df):
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=[df["GDP at current market prices"], df["GDP at basic prices"]], palette="Set3", width = 0.3)
    sns.swarmplot(data=[df["GDP at current market prices"], df["GDP at basic prices"]], color=".25")
    plt.xticks([0, 1], ['GDP(market)', 'GDP(basic)']) #label the group names in x-axis
    fig = plt.gcf()
    fig_size = fig.get_size_inches()  # Get current size
    sizefactor = 1.2  # Set a zoom factor
    # Modify the current size by the factor
    fig.set_size_inches(sizefactor * fig_size)
    fig.savefig('boxplot_GDP.png')#save the fig
    return fig


@st.cache(allow_output_mutation=True)
def piechart_dist(): #by activity
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize =(20, 17))
    x = ['Manufacturing','Envrionment','Construction','Service']
    y1 = [61399, 37671, 62532, 1120265]
    y2 = [30410, 34486, 56531, 1614922]
    y3 = [25140, 35155, 104017, 2397020]

    ax[0].pie(y1, labels=x, radius = 1.3)
    ax[1].pie(y2, labels=x, radius = 1.3)
    ax[2].pie(y3, labels=x, radius = 1.3)

    ax[0].set_title('2000')
    ax[1].set_title('2010')
    ax[2].set_title('2020')

    #plt.legend()
    title = "GDP by activity in 2000, 2010, 2020"
    fig = plt.gcf()
    fig_size = fig.get_size_inches()  # Get current size
    sizefactor = 0.7 # Set a zoom factor
    # Modify the current size by the factor
    fig.set_size_inches(sizefactor * fig_size)
    fig.savefig('pie_GDP.png')  # save the fig
    return fig


def Uber_dataset():
    #Uber-raw-data-apr14 dataset
    st.title("Uber data visualization")

    ## Load the Data
    st.text(" ")
    st.header("Load the Data")
    st.text(" ")
    df1 = load_data(uber_path)
    if st.checkbox('Show dataframe'):
        df1

    ## Perform Data Transformation
    df1_ = df1_data_transformation(df1)

    ## Visual representation
    st.text(" ")
    st.text(" ")
    st.header("Visual representation")
    if st.checkbox("Show graphs"):
        st.text(" ")
        st.markdown("`Frequency by day of the month`")
        st.pyplot(frequency_by_dom(df1_))

        #
        st.text(" ")
        st.markdown("`Viewing points on a map`")
        st.map(map_data(df1_))

        #
        st.text(" ")
        st.markdown("`Visualization of data per hour`")
        st.pyplot(data_by("hours",df1_))

        #
        st.text(" ")
        st.markdown("`Visualization of data by day of the week`")
        st.pyplot(data_by("dow",df1_))

        #
        st.text(" ")
        st.markdown("`Visualization of data by day of the week with the names of the days in abscissa`")
        st.pyplot(data_by("dow_xticks",df1_))

        #
        st.text(" ")
        st.markdown("`Mean latitude and longitude by day of the month`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(data_by("dom",df1_))

    st.text(" ")
    st.text(" ")
    ## Performing Cross Analysis
    st.header("Performing Cross Analysis")

    if st.checkbox('Show cross analysis'):

        #
        grp_df = group_by_wd(df1_)

        #
        st.text(" ")
        st.markdown("`Heatmap with grouped data`")
        st.pyplot(grp_heatmap(grp_df))

        #
        st.text(" ")
        st.markdown("`Histogram of latitude and longitude`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(lat_lon_hist(df1_))

        #
        st.text(" ")
        st.markdown("`Merges latitude and longitude histograms`")
        st.pyplot(lat_lon_hist(df1_, fusion=True))

        #
        st.text(" ")
        st.markdown("`Display of the latitude points on a graph`")
        st.pyplot(display_points(df1_.Lat))

        #
        st.text(" ")
        st.markdown("`Display of the longitude points on a graph`")
        st.pyplot(display_points(df1_.Lon, color="orange"))

def Ny_dataset():
    #ny-trips-data dataset
    st.title("New York taxi trips")

    ## Load the Data
    st.text(" ")
    st.header("Load the Data")
    st.text(" ")
    df2 = load_data(ny_path)
    if st.checkbox('Show dataframe'):
        df2

    ## Perform Data Transformation
    st.text(" ")
    df2_ = df2_data_transformation(df2)

    ## Visual representation
    st.text(" ")
    st.text(" ")
    st.header("Visual representation")
    if st.checkbox('Show graphs'):
        st.text(" ")
        st.text(" ")
        st.markdown("`Total number, average number of passengers and total number of crossings per departure time`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(passengers_graphs_per_hour(df2_))

        #
        st.text(" ")
        st.text(" ")
        st.markdown("`Total number, average number of passengers and total number of crossings per arrival time`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(passengers_graphs_per_dropoff_hour(df2_))

        #
        st.text(" ")
        st.text(" ")
        st.markdown("`Total amount and average amount collected according to departure time`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(amount_graphs_per_hour(df2_))

        #
        st.text(" ")
        st.text(" ")
        st.markdown("`Total distance traveled and average distance according to departure time`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(distance_graphs_per_hour(df2_))

    ## Performing Cross Analysis
    st.text(" ")
    st.text(" ")
    st.header("Performing Cross Analysis")
    if st.checkbox('Show cross analysis'):
        st.text(" ")
        st.markdown("`Heat map to visualize the correlation between the different features`")
        st.pyplot(corr_heatmap(df2_.corr()))

        #
        st.text(" ")
        st.markdown("`Heatmap showing the correlation between number of passengers, total distance, amount of fare, tip and total amount grouped by departure time`")
        grp = df2_[["passenger_count", "hours_pickup", "trip_distance", "fare_amount", "tip_amount", "total_amount"]].groupby("hours_pickup").sum()
        st.pyplot(corr_heatmap(grp.corr()))

def netflix_data():
    #Netflix dataset
    st.title("Netflix Movies and TV shows")

    ## Load the Data
    st.text(" ")
    st.header("Load the Data")
    st.text(" ")
    df3 = load_data(netflix)
    if st.checkbox('Show dataframe'):
        df3

def hk_GDP_data():
    # GDP data
    st.title("Dat Visualization on Gross Domestic Product (GDP) by major economic activity at current prices, Hong Kong (2000-2020)")
    st.subheader("by Dr. Mingyu WAN, Clara")
    st.subheader("Created on 27 April 2022")

    ## Load the Data
    st.text(" ")
    st.header("Load the Data")
    st.text(" ")
    df4 = load_data(gdp_path)
    if st.checkbox('Show dataframe'):
        st.markdown("Data downloaded from DATA.GOV.HK: https://data.gov.hk/en-data/dataset/hk-censtatd-tablechart-gdp-by-economic-activity.")
        df4
        st.markdown("Source: National Income Section(2)1, Census and Statistics Department")

## Visual representation
    st.text(" ")
    st.text(" ")
    st.header("Visual representation")
    #overall trend of GDP by current market price
    if st.checkbox("Yearly Distribution"):
        st.text(" ")
        st.markdown("`Amount by million HK dollars`")
        st.pyplot(lineplot_trend(df4))
        st.markdown("`Basically a climbing trend until a peak at Year 2017 and a decline in 2020.`")


    #distribution compared to GDP by basic price
    if st.checkbox("Compare with GDP at basic prices"):
        st.text(" ")
        st.markdown("`Distribution samples based on years (2000-2020)`")
        st.pyplot(boxplot_dist(df4))
        st.markdown("`Similar distribution, GDP at market prices is a bit higher than at basic prices.`")

    #distribution by activity
    if st.checkbox("GDP by economic activity"):
        st.text(" ")
        st.markdown("`Compare the distribution between year 2000, 2010, 2020`")
        st.pyplot(piechart_dist())
        st.markdown("`Service is the most active economy, and it is increasing yearly; "
                    "in contrast, Manufacturing and Environment are decreasing in GDP; "
                    "Note that Agriculture is too small to be seen in data.`")

## Correlation Analysis
    st.text(" ")
    st.text(" ")
    st.header("Correlation Analysis")

    #correlation matrix
    if st.checkbox("correlation matrix"):
        st.text(" ")
        st.markdown("`Correlation among variables of the economical activies, tax and GDP`")
        df5 = df4[["Agriculture", "Manufacturing", "Environment", "Construction", 'Services', "Taxes on products", "GDP at current market prices"]]
        st.pyplot(corr_heatmap(df5))
        st.markdown("`It seems GDP negatively correlates to Environment and Manufacturing.`")

def hk_export_data():
    # export data
    st.title("Dat Visualization on Total exports by Main Destinations of Hong Kong in 2020")
    st.subheader("by Dr. Mingyu WAN, Clara")
    st.subheader("Created on 27 April 2022")

    ## Load the Data
    st.text(" ")
    st.header("Load the Data")
    st.text(" ")
    df_export = load_data(export_path)
    if st.checkbox('Show dataframe'):
        st.markdown("Data downloaded from Census and Statistics Department, Trade and Cargo Statistics: "
                    "https://tradeidds.censtatd.gov.hk/Index/d93abfca65844f8eb0d676cf631c4821"
                )
        df_export
        st.markdown("Source: Trade Analysis Section (2), Census and Statistics Department")

    #world map of export from hk
    if st.checkbox("world map"):
        st.text(" ")
        st.markdown("`Total exports by Main Destinations (countries) in 2020`")

        # create a world map
        worldmap = pygal.maps.world.World()

        # set the title of the map
        worldmap.title = 'Countries'

        # adding the countries
        worldmap.add('HK Export Data', {
            'cn': 241661,
            'us': 23443,
            'tw': 10463,
            'in': 7930,
            'jp': 10113,
            'vn': 8139,
            'sg': 5070,
            'ae': 5591,
            'kr': 5208,
            'de': 5563
        })
        worldmap.render()
        fig = plt.gcf()
        fig.savefig('export_wm.png')  # save the fig
        # save into the file
        worldmap.render_to_file('abc.svg')
        st.pyplot(fig)

        st.markdown("`The biggest export country: China, followed by USA and Taiwan`")

def main():

    choice = st.sidebar.selectbox(
    'Choose your dataset',
    ('HK GDP by activity dataset', 'HK export data', 'Uber NYC dataset', 'NYC taxi trip dataset', 'Netflix dataset'))

    if choice == 'Uber NYC dataset':
        Uber_dataset()
    elif choice == 'NYC taxi trip dataset':
        Ny_dataset()
    elif choice == 'HK GDP by activity dataset':
        hk_GDP_data()
    elif choice == 'HK export data':
        hk_export_data()
    else:
        netflix_data()

main()