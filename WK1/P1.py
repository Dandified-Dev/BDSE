import pandas as pd
import matplotlib.pyplot as plt


#Read the data into a dataframe
df = pd.read_csv("./data/football_results.csv")

#Print first rows of the dataframe
print(df.head())

#Unqiue home teams
print(df["home"].nunique())

#Unique away teams
print(df["away"].nunique())

#Turn home and away into sets
home_teams = set(df['home'])
away_teams = set(df['away'])

#Number of teams mentioned in away, but never in the home
print(away_teams - home_teams)
print(len(away_teams - home_teams))

#Wins, draws and loses
df['home_wins'] = df.apply(lambda row: 1 if row['gh'] > row['ga'] else 0, axis=1)
df['home_loses'] = df.apply(lambda row: 1 if row['gh'] < row['ga'] else 0, axis=1)
df['home_draws'] = df.apply(lambda row: 1 if row['gh'] == row['ga'] else 0, axis=1)

print(df[["home","away", "gh","ga" ,"home_wins", "home_draws", "home_loses"]].head())

#Sum of homewins and sorted
home_stats = df.groupby("home").agg(home_wins=('home_wins', 'sum'),
                                    home_draws=('home_draws', 'sum'),
                                    home_loses=('home_loses', 'sum'),
                                    total_games=('home_wins', 'count')).reset_index()

min_value = 500
head_value = 10

home_stats = home_stats[home_stats['total_games'] >= min_value]
home_stats['win_rate'] = home_stats['home_wins'] / home_stats['total_games']
home_stats['draw_rate'] = home_stats['home_draws'] / home_stats['total_games']
home_stats['lose_rate'] = home_stats['home_loses'] / home_stats['total_games']

print(home_stats.sort_values("win_rate", ascending=False).head(head_value))

#Making and showing the plot of rates
rate_plot = home_stats.sort_values("win_rate", ascending=False).head(head_value).plot(kind="bar", x="home", y=["win_rate", "draw_rate", "lose_rate"], stacked=True, xlabel="Team", ylabel="Percentage")

rate_plot.legend(["Perc_home_wins", "Perc_home_draws", "Perc_home_loses"])
plt.tight_layout()
plt.show()
