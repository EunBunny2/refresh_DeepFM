
import pandas as pd


'''
 uid : user id
 mid : movie id
 rating : user가 매긴 평점
 timestamp : 시간대?

 movie_genre : |로 구분되어 여러 개의 값이 들어있음 (multi level)

 user_fea1 : sex
 user_fea2 ~ user_fea4 : 관련 tag

 ratings.dat : 'uid', 'mid', 'rating', 'timestamp'로 이루어진 파일 
 movies.dat : 'mid', 'movie_name', 'movie_genre'로 이루어진 파일
 users.dat : 'uid', 'user_fea1', 'user_fea2', 'user_fea3', 'user_fea4'로 이루어진 파일
 

'''


def load_input():
    COL_NAME = ['mlsfc', 'mcate_nm', 'Sex', 'Age', 'Month', 'time', 'day', 'fav_plc', 'click']
    df = pd.read_csv('dataset/new_input.csv', sep=',', header=0, engine='python', names=COL_NAME, encoding='utf-8-sig')
    return df, COL_NAME

'''
	uid	mid	rating	timestamp	movie_name	movie_genre	user_fea1	user_fea2	user_fea3	user_fea4
0	1	1193	5	978300760	One Flew Over the Cuckoo's Nest (1975)	[1, 0, 0]	F	1	10	48067
1	1	661	3	978302109	James and the Giant Peach (1996)	[9, 13, 0]	F	1	10	48067
2	1	914	3	978301968	My Fair Lady (1964)	[13, 5, 0]	F	1	10	48067
3	1	3408	4	978300275	Erin Brockovich (2000)	[1, 0, 0]	F	1	10	48067
4	1	2355	5	978824291	Bug's Life, A (1998)	[9, 2, 0]	F	1	10	48067
'''