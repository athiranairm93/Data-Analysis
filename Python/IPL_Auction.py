#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


ipl_df = pd.read_csv("D:/Data Science/My_Projects/IPL/IPLData.csv")
ipl_df.head()


# In[4]:


#We can check total no: of null values using  the isna().sum() method
ipl_df.isna().sum()


# In[5]:


ipl_df.info()


# In[6]:


# Batters that have played already IPL
batters = ipl_df.loc[(ipl_df['Player_Type'] == "Batter")]
capped_batters = batters.loc[(batters['Capped'] == 1)]
# Table with Batters that have already played IPL
capped_batters_df = capped_batters[['Player Name','Team','Nationality','Matches_Played','Runs','Average','Strike_Rate']]
capped_batters_df.head()


# In[7]:


#Experienced Bowlers
bowlers = ipl_df.loc[(ipl_df['Player_Type'] == "Bowler ")]
capped_bowlers = bowlers.loc[(bowlers['Capped'] == 1)]
# Table of experienced bowlers
capped_bowlers_df = capped_bowlers[['Player Name','Team','Nationality',
                                    'Matches_Played','Wickets','Bowling_average','Bowling_Strike_Rate','Economy']]
capped_bowlers_df.head()


# In[8]:


#Experienced Keepers
keepers = ipl_df.loc[(ipl_df['Player_Type'] == "Keeper")]
capped_keepers = keepers.loc[(keepers['Capped'] == 1)]
# Table of experienced Keepers
capped_keepers_df = capped_keepers[['Player Name','Team','Nationality',
                                    'Matches_Played','Runs','Average','Strike_Rate','Catches','Run_outs','Stumps']]
capped_keepers_df.head()


# In[9]:


#Experienced Allrounders
allrounders = ipl_df.loc[(ipl_df['Player_Type'] == "Allrounder")]
capped_allrounders = allrounders.loc[(allrounders['Capped'] == 1)]
# Table of experienced Allrounders
capped_allrounders_df = capped_allrounders[['Player Name','Team','Nationality',
                                    'Matches_Played','Runs','Average','Strike_Rate','Wickets','Bowling_average','Bowling_Strike_Rate','Economy']]
capped_allrounders_df.head()


# In[10]:


#Cleaning the data by making NAN or Null values to 0
capped_batters_df = capped_batters_df.fillna(0)
capped_bowlers_df = capped_bowlers_df.fillna(0)
capped_keepers_df = capped_keepers_df.fillna(0)
capped_allrounders_df = capped_allrounders_df.fillna(0)


# In[11]:


print(capped_batters_df.isna().sum())
print(capped_bowlers_df.isna().sum())
print(capped_keepers_df.isna().sum())
print(capped_allrounders.isna().sum())


# In[12]:


#Initial analysis
# we analyse data for batters, bowlers, keepers and allrounders

# Analyzing batter data
# Here we have narrowed our analsis to batters who have a batting avg more than 32
top_batters = capped_batters_df.loc[(capped_batters_df['Average'] > 32.0)]
#Sorting data in descending w.r.t each parameter
top_batters_avg = top_batters.sort_values('Average', ascending = False)
top_batters_strike_rate = top_batters.sort_values('Strike_Rate', ascending = False)
top_batters_strike_runs = top_batters.sort_values('Runs', ascending = False)
top_batters_matches = top_batters.sort_values('Matches_Played', ascending = False)


# In[13]:


#data of each batters in descending order of batting avg
top_batters_avg


# In[14]:


#data of each batters in descending order of Strike rate
top_batters_strike_rate


# In[16]:


#data of each batters in descending order of Run rates
top_batters_strike_runs


# In[17]:


#data of each batters in descending order of matches Playes
top_batters_matches


# In[33]:


#From our analysis the top 3 batters taht will come while analysis each of the above data are:
#David Warner
#KL Rahul
#Virat Kohli


# In[18]:


# we analyse data for batters, bowlers, keepers and allrounders

# Analyzing Bowlers data
# Here we have narrowed our analsis based on the bowling avg of the players to be less than 24
top_bowlers = capped_bowlers_df.loc[(capped_bowlers_df['Bowling_average'] < 24.0)]

#Sorting data w.r.t each parameter
top_bowlers_avg = top_bowlers.sort_values('Bowling_average')
top_bowlers_strike_rate = top_bowlers.sort_values('Bowling_Strike_Rate')
top_bowlers_wickets = top_bowlers.sort_values('Wickets', ascending = False)
top_bowlers_economy = top_bowlers.sort_values('Economy')
top_bowlers_matches = top_bowlers.sort_values('Matches_Played', ascending = False)


# In[19]:


#data of each bowlers based on their bowling avg (Top bowlers)
top_bowlers_avg


# In[20]:


#data of each bowlers based on their bowling strike rate (Top bowlers)
top_bowlers_strike_rate


# In[21]:


#Bowlers with highest wickets
top_bowlers_wickets


# In[22]:


top_bowlers_economy


# In[23]:


#Largest no of matches played
top_bowlers_matches


# In[40]:


#From our analysis the top 3 bowlers taht will come while analysis each of the above data are:
#Kagiso Rabada 
#Jasprit Bumrah
#Yuzvendra Chahal
#Nathan Coulter-Nile


# In[24]:


# we analyse data for batters, bowlers, keepers and allrounders

# Analyzing allrounders data
# Here we have narrowed our analsis based on the strike rate of the players to be greater than or equal to 140
top_allrounders = capped_allrounders_df.loc[(capped_allrounders_df['Strike_Rate'] >= 140.0)]

#Sorting data w.r.t each parameter
top_allrounders_batters_avg = top_allrounders.sort_values('Average', ascending = False)
top_allrounders_batters_strike_rate = top_allrounders.sort_values('Strike_Rate', ascending = False)
top_allrounders_runs = top_allrounders.sort_values('Runs', ascending = False)
top_allrounders_matches = top_allrounders.sort_values('Matches_Played', ascending = False)
top_allrounders_bowlers_avg = top_allrounders.sort_values('Bowling_average')
top_allrounders_bowlers_strike_rate = top_allrounders.sort_values('Bowling_Strike_Rate')
top_allrounders_wickets = top_allrounders.sort_values('Wickets', ascending = False)
top_allrounders_economy = top_allrounders.sort_values('Economy')


# In[25]:


# Top allrounders based on their batting avg
top_allrounders_batters_avg


# In[26]:


# Top allrounders based on their batting strike rate
top_allrounders_batters_strike_rate


# In[27]:


# Top allrounders with highest runs
top_allrounders_runs


# In[28]:


# Top allrounders with highest matches
top_allrounders_matches


# In[29]:


# Top allrounders based on bowing avg
top_allrounders_bowlers_avg


# In[30]:


# Top allrounders based on bowing strike rate
top_allrounders_bowlers_strike_rate


# In[31]:


# Top allrounders with highest wickets
top_allrounders_wickets


# In[32]:


# Top allrounders with best economy
top_allrounders_economy


# In[33]:


#Andre Russell 
#Sunil Narine 
#Hardik Pandya 
#Jofra Archer


# In[35]:


# we analyse data for batters, bowlers, keepers and allrounders
#Keepers avg should be greater than or equal 25
top_keepers = capped_keepers_df.loc[(capped_keepers_df['Average'] >= 25.0)]
#Sorting
top_keepers_avg = top_keepers.sort_values('Average', ascending = False)
top_keepers_runs = top_keepers.sort_values('Runs', ascending = False)
top_keepers_matches_played = top_keepers.sort_values('Matches_Played', ascending = False)
top_keepers_strike_rate = top_keepers.sort_values('Strike_Rate', ascending = False)
top_keepers_catches = top_keepers.sort_values('Catches', ascending = False)
top_keepers_run_outs = top_keepers.sort_values('Run_outs', ascending = False)
top_keepers_stumps = top_keepers.sort_values('Stumps', ascending = False)


# In[36]:


top_keepers_avg


# In[37]:


top_keepers_runs


# In[38]:


top_keepers_matches_played


# In[39]:


top_keepers_strike_rate


# In[40]:


top_keepers_catches


# In[41]:


top_keepers_run_outs


# In[42]:


top_keepers_stumps


# In[43]:


#MS Dhoni
#Dinesh Karthik
#Rishabh Pant


# In[44]:


#Visuliztion for enhanced Analysis


# In[53]:


#Top batters based on their strike rate
plt.figure(figsize=(20,10))
plt.title("Top best batters w.r.t Strike Rate")
sns.barplot(data=top_batters_strike_rate, x = 'Player Name', y = 'Strike_Rate')


# In[55]:


#Top batters based on their Runs
plt.figure(figsize=(20,10))
plt.title("Top best batters with highest runs")
sns.barplot(data=top_batters_strike_runs, x = 'Player Name', y = 'Runs')


# In[56]:


#Top batters based on their Average
plt.figure(figsize=(20,10))
plt.title("Top best batters with highest Average")
sns.barplot(data=top_batters_avg, x = 'Player Name', y = 'Average')


# In[57]:


#Top batters based on No of matches played
plt.figure(figsize=(20,10))
plt.title("Top best batters with more matches played")
sns.barplot(data=top_batters_matches, x = 'Player Name', y = 'Matches_Played')


# In[58]:


#Top Bowlers based on their bowling avg
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their bowling avg")
sns.barplot(data=top_bowlers_avg, x = 'Player Name', y = 'Bowling_average')


# In[61]:


#Top Bowlers based on their bowling strike rates
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their strike rates")
sns.barplot(data=top_bowlers_avg, x = 'Player Name', y = 'Bowling_Strike_Rate')


# In[63]:


#Top Bowlers based on their wickets
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their wickets")
sns.barplot(data=top_bowlers_wickets, x = 'Player Name', y = 'Wickets')


# In[64]:


#Top Bowlers based on their Economy
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their Economy")
sns.barplot(data=top_bowlers_economy, x = 'Player Name', y = 'Economy')


# In[66]:


#Top Bowlers based on Matches Played
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their Matches")
sns.barplot(data=top_bowlers_matches, x = 'Player Name', y = 'Matches_Played')


# In[67]:


#Top AllRounders based on Batting Average
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their Batting Average")
sns.barplot(data=top_allrounders_batters_avg, x = 'Player Name', y = 'Average')


# In[68]:


#Top AllRounders based on Bowling Average
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their Bowling Average")
sns.barplot(data=top_allrounders_bowlers_avg, x = 'Player Name', y = 'Bowling_average')


# In[69]:


#Top AllRounders based on Batters Strike Rate
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their Batters Strike Rate")
sns.barplot(data=top_allrounders_batters_strike_rate, x = 'Player Name', y = 'Strike_Rate')


# In[70]:


#Top AllRounders based on Bowlers Strike Rate
plt.figure(figsize=(20,10))
plt.title("Top best Bowlers based on their Bowlers Strike Rate")
sns.barplot(data=top_allrounders_bowlers_strike_rate, x = 'Player Name', y = 'Bowling_Strike_Rate')


# In[71]:


#Top AllRounders based on Runs
plt.figure(figsize=(20,10))
plt.title("Top best Players by Runs")
sns.barplot(data=top_allrounders_runs, x = 'Player Name', y = 'Runs')


# In[72]:


#Top AllRounders based on Matches Played
plt.figure(figsize=(20,10))
plt.title("Top best Players by Matches Played")
sns.barplot(data=top_allrounders_matches, x = 'Player Name', y = 'Matches_Played')


# In[73]:


#Top AllRounders based on Wickets
plt.figure(figsize=(20,10))
plt.title("Top best Players by Wickets")
sns.barplot(data=top_allrounders_wickets, x = 'Player Name', y = 'Wickets')


# In[74]:


#Top AllRounders based on Economy
plt.figure(figsize=(20,10))
plt.title("Top best Players by Economy")
sns.barplot(data=top_allrounders_economy, x = 'Player Name', y = 'Economy')


# In[75]:


#Top Keepers based on Average
plt.figure(figsize=(20,10))
plt.title("Top best Keepers by Average")
sns.barplot(data=top_keepers_avg, x = 'Player Name', y = 'Average')


# In[76]:


#Top Keepers based on Runs
plt.figure(figsize=(20,10))
plt.title("Top best Keepers by Runs")
sns.barplot(data=top_keepers_runs, x = 'Player Name', y = 'Runs')


# In[77]:


#Top Keepers based on Matches Played
plt.figure(figsize=(20,10))
plt.title("Top best Keepers by Matches Played")
sns.barplot(data=top_keepers_matches_played, x = 'Player Name', y = 'Matches_Played')


# In[78]:


#Top Keepers based on Strike Rate
plt.figure(figsize=(20,10))
plt.title("Top best Keepers by Strike Rate")
sns.barplot(data=top_keepers_strike_rate, x = 'Player Name', y = 'Strike_Rate')


# In[79]:


#Top Keepers based on Strike Catches
plt.figure(figsize=(20,10))
plt.title("Top best Keepers by Catches")
sns.barplot(data=top_keepers_catches, x = 'Player Name', y = 'Catches')


# In[80]:


#Top Keepers based on Run Outs
plt.figure(figsize=(20,10))
plt.title("Top best Keepers by Run Outs")
sns.barplot(data=top_keepers_run_outs, x = 'Player Name', y = 'Run_outs')


# In[81]:


#Top Keepers based on Stumps
plt.figure(figsize=(20,10))
plt.title("Top best Keepers by Stumps")
sns.barplot(data=top_keepers_stumps, x = 'Player Name', y = 'Stumps')


# In[82]:


#For our analysis we consider 3 batters and 3 allrounders, 4 bowlers with 2 spin optns 1 wicket keeper (team of 11)


# In[91]:


#Batters for final 11 - KL Rahul, Virat Kholi, David Warner (based on visualization)
#here we are sorting values of each player in a separate dataframe to use for displaying using barplot
top_batters.reset_index(drop=True)
total_matches = [top_batters.iloc[6]['Matches_Played'],top_batters.iloc[2]['Matches_Played'],top_batters.iloc[5]['Matches_Played']] 
total_runs = [top_batters.iloc[6]['Runs'],top_batters.iloc[2]['Runs'],top_batters.iloc[5]['Runs']]
total_average = [top_batters.iloc[6]['Average'],top_batters.iloc[2]['Average'],top_batters.iloc[5]['Average']]
total_strike_rate = [top_batters.iloc[6]['Strike_Rate'],top_batters.iloc[2]['Strike_Rate'],top_batters.iloc[5]['Strike_Rate']]

Labels = ['KL Rahul','David Warner','Virat Kohli']
fig, axes = plt.subplots(2,2, figsize=(10,10))
axes[0][0].set_title("Matches Played")
axes[0][1].set_title("Runs in IPL Career")
axes[1][0].set_title("Average")
axes[1][1].set_title("Strike Rate")
sns.barplot(x=Labels, y=total_matches, ax=axes[0][0])
sns.barplot(x=Labels, y=total_runs, ax=axes[0][1])
sns.barplot(x=Labels, y=total_average, ax=axes[1][0])
sns.barplot(x=Labels, y=total_strike_rate, ax=axes[1][1])


# In[100]:


#Allrounders for final 11 - Andre Russell, Sunil Narine, Hardik Pandya (based on visualization)
#here we are sorting values of each player in a separate dataframe to use for displaying using barplot
top_allrounders.reset_index(drop=True)
tot_matches = [top_allrounders.iloc[5]['Matches_Played'],top_allrounders.iloc[9]['Matches_Played'],top_allrounders.iloc[6]['Matches_Played']]
tot_runs = [top_allrounders.iloc[5]['Runs'],top_allrounders.iloc[9]['Runs'],top_allrounders.iloc[6]['Runs']]
tot_avg = [top_allrounders.iloc[5]['Average'],top_allrounders.iloc[9]['Average'],top_allrounders.iloc[6]['Average']]
tot_sr = [top_allrounders.iloc[5]['Strike_Rate'],top_allrounders.iloc[9]['Strike_Rate'],top_allrounders.iloc[6]['Strike_Rate']]
tot_wickets = [top_allrounders.iloc[5]['Wickets'],top_allrounders.iloc[9]['Wickets'],top_allrounders.iloc[6]['Wickets']]
tot_bavg = [top_allrounders.iloc[5]['Bowling_average'],top_allrounders.iloc[9]['Bowling_average'],top_allrounders.iloc[6]['Bowling_average']]
tot_bosr = [top_allrounders.iloc[5]['Bowling_Strike_Rate'],top_allrounders.iloc[9]['Bowling_Strike_Rate'],top_allrounders.iloc[6]['Bowling_Strike_Rate']]
tot_economy = [top_allrounders.iloc[5]['Economy'],top_allrounders.iloc[9]['Economy'],top_allrounders.iloc[6]['Economy']]

Labels = ['Andre Russell', 'Sunil Narine', 'Hardik Pandya']
fig, axes = plt.subplots(4,2, figsize=(20,20))
axes[0][0].set_title("Matches Played")
axes[0][1].set_title("Runs")
axes[1][0].set_title("Average")
axes[1][1].set_title("Strike Rate")
axes[2][0].set_title("Wickets")
axes[2][1].set_title("Bowling Average")
axes[3][0].set_title("Bowling Strike Rate")
axes[3][1].set_title("Economy")

sns.barplot(x=Labels, y=tot_matches, ax=axes[0][0])
sns.barplot(x=Labels, y=tot_runs, ax=axes[0][1])
sns.barplot(x=Labels, y=tot_avg, ax=axes[1][0])
sns.barplot(x=Labels, y=tot_sr, ax=axes[1][1])
sns.barplot(x=Labels, y=tot_wickets, ax=axes[2][0])
sns.barplot(x=Labels, y=tot_bavg, ax=axes[2][1])
sns.barplot(x=Labels, y=tot_bosr, ax=axes[3][0])
sns.barplot(x=Labels, y=tot_economy, ax=axes[3][1])


# In[107]:


#Bowlers for final 11 - Jasprit Bumrah, Kagiso Rabada, Nathan Coulter-Nile and Yuzvendra Chahal (based on visualization)
#here we are sorting values of each player in a separate dataframe to use for displaying using barplot
10,0,7,1
top_bowlers


# In[120]:


top_bowlers.reset_index(drop=True)
bowlers_matches = [top_bowlers.iloc[10]['Matches_Played'],top_bowlers.iloc[0]['Matches_Played'],top_bowlers.iloc[7]['Matches_Played'],top_bowlers.iloc[1]['Matches_Played']]
bowlers_wickets = [top_bowlers.iloc[10]['Wickets'],top_bowlers.iloc[0]['Wickets'],top_bowlers.iloc[7]['Wickets'],top_bowlers.iloc[1]['Wickets']]
bowlers_average = [top_bowlers.iloc[10]['Bowling_average'],top_bowlers.iloc[0]['Bowling_average'],top_bowlers.iloc[7]['Bowling_average'],top_bowlers.iloc[1]['Bowling_average']]
bowlers_strike_rate = [top_bowlers.iloc[10]['Bowling_Strike_Rate'],top_bowlers.iloc[0]['Bowling_Strike_Rate'],top_bowlers.iloc[7]['Bowling_Strike_Rate'],top_bowlers.iloc[1]['Bowling_Strike_Rate']]
bowlers_economy = [top_bowlers.iloc[10]['Economy'],top_bowlers.iloc[0]['Economy'],top_bowlers.iloc[7]['Economy'],top_bowlers.iloc[1]['Economy']]

Labels = ['Jasprit Bumrah','Kagiso Rabada','Nathan Coulter-Nile','Yuzvendra Chahal']
fig, axes = plt.subplots(3,2, figsize=(15,15))
axes[0][0].set_title("Matches Played")
axes[0][1].set_title("Top Wicketers")
axes[1][0].set_title("Bowling Average")
axes[1][1].set_title("Bowling Strike Rate")
axes[2][1].set_title("Economy")
sns.barplot(x=Labels, y=bowlers_matches, ax=axes[0][0])
sns.barplot(x=Labels, y=bowlers_wickets, ax=axes[0][1])
sns.barplot(x=Labels, y=bowlers_average, ax=axes[1][0])
sns.barplot(x=Labels, y=bowlers_strike_rate, ax=axes[1][1])
sns.barplot(x=Labels, y=bowlers_economy, ax=axes[2][1])


# In[127]:


#wicket Keeper for final 11 - M.S Dhoni
#here we are sorting values of each player in a separate dataframe to use for displaying using barplot
matches_values = [top_keepers.iloc[8]['Matches_Played'],top_keepers.iloc[8]['Runs']]
average_values = [top_keepers.iloc[8]['Average'],top_keepers.iloc[8]['Strike_Rate']]
keeping_values = [top_keepers.iloc[8]['Catches'],top_keepers.iloc[8]['Stumps'],top_keepers.iloc[8]['Run_outs']]

label1 = ['Matches','Runs']
label2 = ['Average','Strike Rate']
label3 = ['Catches','Stumps', 'Run Outs']
                                               
fig, axes = plt.subplots(1,3, figsize=(20,10))
axes[0].set_title("Matches and Runs")
axes[1].set_title("Average and Strike Rate")
axes[2].set_title("Keeping status")
                                                  
sns.barplot(x=label1, y=matches_values, ax=axes[0])
sns.barplot(x=label2, y=average_values, ax=axes[1])
sns.barplot(x=label3, y=keeping_values, ax=axes[2])


# In[151]:


#Final 11 for IPL
batter1 = top_batters.loc[(top_batters['Player Name'] == 'KL Rahul ')]
batter2 = top_batters.loc[(top_batters['Player Name'] == 'Virat Kohli')]
batter3 = top_batters.loc[(top_batters['Player Name'] == 'David Warner ')]

bowler1 = top_bowlers.loc[(top_bowlers['Player Name'] == 'Jasprit Bumrah')]
bowler2 = top_bowlers.loc[(top_bowlers['Player Name'] == 'Kagiso Rabada ')]
bowler3 = top_bowlers.loc[(top_bowlers['Player Name'] == 'Nathan Coulter-Nile')]
bowler4 = top_bowlers.loc[(top_bowlers['Player Name'] == 'Yuzvendra Chahal ')]

allrounder1 = top_allrounders.loc[(top_allrounders['Player Name'] == 'Andre Russell')]
allrounder2 = top_allrounders.loc[(top_allrounders['Player Name'] == 'Sunil Narine ')]
allrounder3 = top_allrounders.loc[(top_allrounders['Player Name'] == 'Hardik Pandya')]

wicket_keeper = top_keepers.loc[(top_keepers['Player Name'] == 'MS Dhoni')]

final = [batter1, batter2, batter3, bowler1, bowler2, bowler3, bowler4, allrounder1, allrounder2, allrounder3, wicket_keeper]
final_team = pd.concat(final)
final_team = final_team.drop(labels=['Matches_Played','Runs','Average','Strike_Rate','Wickets','Bowling_average',
                                    'Bowling_Strike_Rate','Economy','Catches','Run_outs','Stumps'], axis=1)
final_team.reset_index(drop=True)


# In[ ]:





# In[ ]:




