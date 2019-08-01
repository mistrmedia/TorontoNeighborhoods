
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup
import requests


# In[2]:


# use Beautiful Soup
source = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup = BeautifulSoup(source, 'lxml') 

# retrieve the column names
table = soup.find('table')
table_header_th = table.find_all('th')

column_names=[]
for element in table_header_th: 
    column_names.append(element.text.strip())

# add each row of the table to the list "list_of_rows"

table_body = table.find('tbody')
table_rows = table_body.find_all('tr')

list_of_rows = []

for tr in table_rows:
    td = tr.find_all('td')
    row = [tr.text for tr in td]
    list_of_rows.append(row)


# In[3]:


# Create df DataFrame with the data
df=pd.DataFrame(list_of_rows, columns=column_names)
df.drop(df.index[0], inplace = True)
df = df.reset_index(drop = True)
df.head()


# In[4]:


# Change the names of the columns
df.rename(columns={'Postcode':'PostalCode','Neighbourhood':'Neighborhood'}, inplace=True)


# In[5]:


# Eliminate blank spaces at the end 
df['PostalCode'] = df['PostalCode'].str.rstrip()
df['Neighborhood'] = df['Neighborhood'].str.rstrip()
df['Borough'] = df['Borough'].str.rstrip()


# In[6]:


# Eliminate rows where Borough == 'Not assigned'
df = df[df.Borough != 'Not assigned']
df.shape


# In[7]:


# dataframe A contains the rows with Neighborhoods concatenated *with same Postal Code*
A=df.groupby('PostalCode')['Neighborhood'].apply(lambda tags:', '.join(tags)).to_frame().reset_index() 
# Remove column Neighborhood from df
df = df.drop(['Neighborhood'], axis=1)

# cut duplicated rows on df 
df.drop_duplicates(inplace = True)

# Join both dataframes on df_new
df_new = pd.merge(df, A, on='PostalCode', how='inner')

# Replace not assigned neighborhoods wiht its borough name
df_new.loc[df_new.Neighborhood == 'Not assigned', 'Neighborhood'] = df_new.loc[df_new.Neighborhood == 'Not assigned'].Borough

df_new.shape


# In[8]:


df_new.head(20)


# In[9]:


# read CSV file
# use the link to the csv file that has the geographical coordinates of each postal code of Toronto.


# In[10]:


get_ipython().system(u"wget -q -O 'Geospatial_Coordinates.csv' https://cocl.us/Geospatial_data")
print('Data downloaded!')


# In[11]:


GeoData_df = pd.read_csv('Geospatial_Coordinates.csv')
GeoData_df.head()


# In[12]:


GeoData_df.shape


# In[13]:


# Change Postal Code to PostalCode
GeoData_df.rename(columns={'Postal Code':'PostalCode'}, inplace=True)

# combine dataframes, df_new and GeoData_df on df_GeoComplete
df_GeoComplete = pd.merge(df_new, GeoData_df, on='PostalCode', how='outer')

df_GeoComplete.head(20)


# In[ ]:


# get venues


# # 1. Dataset

# In[ ]:


# Import libraries


# In[22]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system(u"conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# In[23]:


# Create dataframe with Toronto data including Latitude and Longitude
# Note: working with only boroughs that contain the word Toronto. 

# Select only boroughs that contain the word Toronto on df_Toronto.

df_Toronto = df_GeoComplete[df_GeoComplete['Borough'].str.contains('Toronto')].reset_index()
df_Toronto = df_Toronto.drop(['index'], axis=1)
df_Toronto.head(40)


# In[24]:


neighborhoods = df_Toronto


# In[25]:


#Use geopy library to get the latitude and longitude values of New York City.¶


# In[26]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="Toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[27]:


# Define Foursquare Credentials and Version¶


# In[28]:


CLIENT_ID = 'Q2UEL5ONIRUDOANPJWFF1LTQWXZC05CRRD0KBQVWJMH4Y34E' # your Foursquare ID
CLIENT_SECRET = '4KUXVPO4E4FIS2OFO3DD0PLG0ZSSYN4RTYGT3H1H0ZY03H4T' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version|


# # 2. Explore neighborhoods in Toronto
# 

# In[29]:


# Function to get venues for all the neighborhoods in Toronto
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[30]:


# Run the above function on each neighborhood and create a new dataframe called toronto_venues.
LIMIT = 100
toronto_venues = getNearbyVenues(names=neighborhoods['Neighborhood'],
                                   latitudes=neighborhoods['Latitude'],
                                   longitudes=neighborhoods['Longitude']
                                  )


# In[31]:


# To check the size of the resulting dataframe
print(toronto_venues.shape)
toronto_venues.head()


# In[32]:


# To check how many venues were returned for each neighborhood
toronto_venues.groupby('Neighborhood').count()


# In[33]:


# How many unique categories can be curated from all the returned venues
print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# # 3. Analyze Each Neighborhood
# 

# In[34]:


# There is a venue category named 'Neighborhood', so it doesn't work when applying one hot.
toronto_venues.loc[toronto_venues['Venue Category'] == 'Neighborhood']


# In[35]:


# Replace the Venue Category 'Neighborhood' for 'Neighborhood_cat'
toronto_venues.loc[toronto_venues['Venue Category'] == 'Neighborhood', 'Venue Category'] = 'Neighborhood_cat'

toronto_venues.loc[toronto_venues['Venue Category'] == 'Neighborhood_cat']
#df_new.loc[df_new.Neighborhood == 'Not assigned', 'Neighborhood'] = df_new.loc[df_new.Neighborhood == 'Not assigned'].Borough


# In[36]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")
toronto_onehot.head()


# In[37]:


# add neighborhood column back to dataframe
toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[38]:


# To group rows by neighborhood and by taking the mean of the frequency of occurrence of each category
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped


# In[39]:


# Function to sort the venues in descending order.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[40]:


# To create the new dataframe and display the top 10 venues for each neighborhood.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# # 4. Cluster Neighborhoods

# In[43]:


# To run k-means to cluster the neighborhood into 5 clusters.

# set number of clusters
kclusters = 4

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:50]


# In[44]:


# To create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = neighborhoods

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head(40) # check the last columns!


# In[45]:


# To visualize the resulting clusters
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# # 5. Examine Clusters

# In[46]:


# Cluster 1
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[47]:


# Cluster 2
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[48]:


# Cluster 3
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]


# In[49]:



# Cluster 4
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]

