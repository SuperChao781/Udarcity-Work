# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:00:36 2018

@author: wangyuchao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#数据等级划分
def cutData(df,column_label,new_column_label,cut_list,label_list):
    
    df[new_column_label] =  pd.cut(df[column_label],bins=cut_list,labels=label_list);
    
    return df;
    
#数据标准化
def data_standard(inputdata,ddof_value):
    inputdata_mean = inputdata.mean();
    inputdata_std = inputdata.std(ddof=ddof_value);
    outputdata = (inputdata-inputdata_mean)/inputdata_std;
    return outputdata;


def load_data(filename):
    df = pd.read_csv(filename);
    
    #排除评价次数小于50的电影（样本太小，评分不可信）
    df = df[df['vote_count']>=50];
    #排除 预算/电影收入小于100美元的电影（非法值）
    df = df[df['budget_adj']>100];
    df = df[df['revenue_adj']>100];
    #排除 受欢迎程度和评分为0的电影
    df = df[df['popularity']>0];
    df = df[df['vote_average']>0];
   
    return df;

def analyse_data(df):
    '''**************************************************************'''
    #对数据进行检查
    print(' ')
    print('*********数据检查*********')
    df.info()
     
    '''**************************************************************'''
    print(' ')
    print('*********分析影响电影平均收入的因素*********')
    #调查电影平均收入统计值
    data_describe = df['revenue_adj'].describe()
    #将电影收入划分为一定的级别
    cut_list = [data_describe['min'],1e6,data_describe['25%'],data_describe['50%']
    ,data_describe['75%'],data_describe['max']];
    label_list = ['very low','low','medium','high','very high'];
    df = cutData(df,'revenue_adj','revenue_level',cut_list,label_list);
    #统计各收入级别级别的电影数量
    print(' ')
    print('***对电影收入分级,统计各收入级别的电影数量***')
    print(' ')
    print(df['revenue_level'].value_counts())
    #计算各级别的平均受欢迎程度、平均评分和平均预算
    popularity_mean = df.groupby('revenue_level').mean()['popularity'];
    voter_mean = df.groupby('revenue_level').mean()['vote_average'];
    budget_mean = df.groupby('revenue_level').mean()['budget_adj'];
    
    #用条状图绘制对比图                       
    print(' ')
    print('***绘图显示各收入级别的电影特征***')
    popularity_mean.plot(kind='bar', title='Average popularity by revenue level');
    plt.ylabel('average popularity');
    plt.show()
    voter_mean.plot(kind='bar', title='Average voter count by revenue level');
    plt.ylabel('average voter count');
    plt.show()
    budget_mean.plot(kind='bar', title='Average budget count by revenue level');
    plt.ylabel('average budget');
    plt.show()
    
    
    #将每个电影风格单独列为一行
    df_genres = pd.DataFrame();
    
    #建立一个新数据 储存每个电影的风格及平均受欢迎程度、平均收入

    genres = df['genres'].str.split('|',expand=True).stack().reset_index(level=1,drop=True).rename('genres_separate');
    
    df_genres = df.join(genres);

    #统计所有电影风格
    genres_list = list(df_genres['genres_separate'].value_counts().index);
    
    
    #绘出各个电影风格的收入等级对比
    print(' ')
    print('***绘图显示每种电影风格中各个收入等级数量对比***')
    for genres in genres_list:
        df_genres_revenue = df_genres[df_genres['genres_separate'] == genres];
        df_genres_revenue['revenue_level'].value_counts().plot(kind='pie');       
        plt.title('revenue_level  for genres:'+ genres);
        plt.show()
    
    
    
    '''**************************************************************'''
    #列出净利润最高的前5部电影（电影收入-预算最高，不考虑通货膨胀）
    print(' ')
    print('*********列出净利润最高的5部电影*********')
    film_list = [];
    benefit_list = [];
    benefit=df['revenue']-df['budget'];
    
    for i in range(5):
        film_list.append(df.loc[benefit.argmax()]['original_title']);
        benefit_list.append(benefit.max())
        del(benefit[benefit.argmax()])    
    
    print(' ')
    print('***净利润最高的前5部电影,不考虑通货膨胀***')
    for i in range(len(film_list)):
        print("{0}th on benefit:<{1}>, {2} dollars".format(i,film_list[i],benefit_list[i]))
        
        
    #列出净利润最高的前5部电影（电影收入-预算最高，考虑通货膨胀）
    film_list = [];
    benefit_list = [];
    benefit=df['revenue_adj']-df['budget_adj'];
    
    for i in range(5):
        film_list.append(df.loc[benefit.argmax()]['original_title']);
        benefit_list.append(benefit.max())
        del(benefit[benefit.argmax()])    
    
    print(' ')
    print('***净利润最高的前5部电影,考虑通货膨胀***')
    for i in range(len(film_list)):
        print("{0}th on benefit:<{1}>, {2} dollars".format(i,film_list[i],benefit_list[i]))
    
    '''**************************************************************'''
    #演员统计
    print(' ')
    print('*********演员分析*********')
    df_cast = pd.DataFrame();
    #建立一个新数据 储存每个演员演出的每部电影的受欢迎程度、电影评分和平均收入
    cast = df['cast'].str.split('|',expand=True).stack().reset_index(level=1,drop=True).rename('cast_separate');
    
    df_cast = df.join(cast);
    df_cast.reset_index(inplace=True);

    #分析所有演员出演电影数量
    df_cast_value_count = df_cast['cast_separate'].value_counts();
    #将演员的出演电影数量按5为间隔分为5个等级 如下：
    df_cast_value_count = df_cast_value_count.to_frame();
    cut_list=list(np.arange(0,51,5));#产生分割序列
    cut_label=['{0}_{1}'.format(x+1,x+5) for x in cut_list];#产生分割序列标号
    cut_label.pop();#删掉最后一个
    df_cast_value_count['cast_filmNum']=pd.cut(df_cast_value_count['cast_separate'],bins=cut_list,labels=cut_label);
    #统计各个数量等级的演员数
    print(' ')
    print('***演员参演电影数量分布***')
    df_cast_num_value_count=df_cast_value_count['cast_filmNum'].value_counts();   
    df_cast_num_value_count = df_cast_num_value_count.sort_index();
    df_cast_num_value_count.plot(kind='bar')
    plt.xlabel('the number of movies the actros took part in')
    plt.ylabel('the number of actors who fit the x-axis')
    plt.show()
        
    #筛选出在数据中出演了超过5部电影的演员
    df_cast_value_count = df_cast_value_count[df_cast_value_count['cast_separate']>5];
    
    
    #列出出演电影数量最高的前5名演员
    print(' ')
    print('***出演电影数量最高的前5名演员***')
    for i in range(5):
        print("{0}th on film count:<{1}>, which is {2} ".format(i+1, df_cast_value_count.index[i],\
        df_cast_value_count.iloc[i,0]))


    #从演员数据中删去所有出演电影数量不超过5部的演员数据
    df_cast_morethan_five = pd.DataFrame();
    for i in range(len(df_cast)):
        if(df_cast.loc[i]['cast_separate'] in df_cast_value_count['cast_separate']):
            df_cast_morethan_five = df_cast_morethan_five.append(df_cast.loc[i]);

    #列出出演电影平均评分最高的前5名演员
    cast_average_vote_average = df_cast_morethan_five.groupby('cast_separate').mean()['vote_average'];
    cast_list = [];
    vote_average_list = [];
    for i in range(5):
        cast_list.append(cast_average_vote_average.argmax());
        vote_average_list.append(cast_average_vote_average.max())
        del(cast_average_vote_average[cast_average_vote_average.argmax()])    
    
    print(' ')
    print('出演电影平均评分最高的5名演员:')
    for i in range(len(cast_list)):
        print("{0}th on average vote_average:<{1}>, which is {2} ".format(i,cast_list[i],vote_average_list[i]))

    #绘出这5名演员参演电影的评分
    for i in range(len(cast_list)):
        df_cast_forone = df_cast[df_cast['cast_separate']==cast_list[i]];
        df_cast_forone = df_cast_forone.sort_values(by=['vote_average'],ascending = False)
        df_cast_forone.index = df_cast_forone['original_title'];
        df_cast_forone['vote_average'].plot(kind='bar');
        plt.xlabel('{0}'.format(cast_list[i]));
        plt.ylabel('vote_average');
        plt.show()
    
    #补充 绘出卢克 天行者扮演者和哈利 波特扮演者的数据
    df_cast_forone = df_cast[df_cast['cast_separate']=='Mark Hamill'];
    df_cast_forone = df_cast_forone.sort_values(by=['vote_average'],ascending = False)

    df_cast_forone = df_cast[df_cast['cast_separate']=='Daniel Radcliffe'];
    df_cast_forone = df_cast_forone.sort_values(by=['vote_average'],ascending = False)


    #列出出演电影平均受欢迎程度最高的前5名演员
    cast_average_popularity = df_cast_morethan_five.groupby('cast_separate').mean()['popularity'];
    cast_list = [];
    popularity_list = [];
    for i in range(5):
        cast_list.append(cast_average_popularity.argmax());
        popularity_list.append(cast_average_popularity.max())
        del(cast_average_popularity[cast_average_popularity.argmax()])    
    
    print(' ')
    print('出演电影平均受欢迎最高的5名演员:')
    for i in range(len(cast_list)):
        print("{0}th on average popularity:<{1}>, which is {2}% ".format(i,cast_list[i],popularity_list[i]))

    #绘出这5名演员参演电影的受欢迎程度
    print(' ')
    print('上述5名演员出演的所有电影受欢迎程度:')
    for i in range(len(cast_list)):
        df_cast_forone = df_cast[df_cast['cast_separate']==cast_list[i]];
        df_cast_forone = df_cast_forone.sort_values(by=['popularity'],ascending = False)
        df_cast_forone.index = df_cast_forone['original_title'];
        df_cast_forone['popularity'].plot(kind='bar');
        plt.xlabel('{0}'.format(cast_list[i]));
        plt.ylabel('popularity');
        plt.show()

    #给出莱昂纳多和摩根弗里曼参演电影的受欢迎程度
    print(' ')
    print('莱昂纳多出演的所有电影受欢迎程度:')
    df_cast_forone = df_cast[df_cast['cast_separate']=='Leonardo DiCaprio'];
    df_cast_forone = df_cast_forone.sort_values(by=['popularity'],ascending = False)
    df_cast_forone.index = df_cast_forone['original_title'];
    df_cast_forone['popularity'].plot(kind='bar');
    plt.xlabel('Leonardo DiCaprio');
    plt.ylabel('popularity');
    plt.show()    
    print(' ')
    print('摩根弗里曼出演的所有电影受欢迎程度:')
    df_cast_forone = df_cast[df_cast['cast_separate']=='Morgan Freeman'];
    df_cast_forone = df_cast_forone.sort_values(by=['popularity'],ascending = False)
    df_cast_forone.index = df_cast_forone['original_title'];
    df_cast_forone['popularity'].plot(kind='bar');
    plt.xlabel('Morgan Freeman');
    plt.ylabel('popularity');
    plt.show()    
    
    '''**************************************************************'''
    #统计电影平均预算和收入是否随着发行年代上涨  
    print(' ')
    print('*********电影平均预算和收入随着发行年代变化情况*********')
    #增加“电影年代1”和“电影年代2”列，分析电影平均预算和平均收入随发行年代的变化
    cut_list1=[1950,1976,1986,1996,2006,2016];
    cut_label1=['old',1980,1990,2000,2010]
    cut_list2=[1950,1981,1991,2001,2016];
    cut_label2=['old',1985,1995,2005]

    df['release_decade1']=pd.cut(df['release_year'],bins=cut_list1,labels=cut_label1,right=False);
    df['release_decade2']=pd.cut(df['release_year'],bins=cut_list2,labels=cut_label2,right=False);
    
    budge_average1 = df.groupby('release_decade1').mean()['budget_adj'];
    budge_average2 = df.groupby('release_decade2').mean()['budget_adj'];
   
    
    revenue_average1 = df.groupby('release_decade1').mean()['revenue_adj'];
    revenue_average2 = df.groupby('release_decade2').mean()['revenue_adj'];
    
    #组合两组数据
    del(budge_average1['old']);
    del(budge_average2['old']);
    del(revenue_average1['old']);
    del(revenue_average2['old']);
    budge_average = pd.Series();
    budge_average=budge_average.append(budge_average1);
    budge_average=budge_average.append(budge_average2);
    budge_average=budge_average.sort_index()
    
    revenue_average = pd.Series();
    revenue_average=revenue_average.append(revenue_average1);
    revenue_average=revenue_average.append(revenue_average2);
    revenue_average=revenue_average.sort_index()
    #绘图
    plt.plot(budge_average)
    plt.xlabel('decade');
    plt.ylabel('budget_adj');
    plt.title('budget change by decades')
    plt.show()
    plt.plot(revenue_average)
    plt.xlabel('decade');
    plt.ylabel('revenue_adj');
    plt.title('revenue change by decades')
    plt.show()
    
    
    
df = pd.DataFrame()

def main():
    global df
    df = load_data('data1_TMDb_film\\tmdb-movies.csv');#导入数据
    
    analyse_data(df);#分析数据
    
    
#    keys = input('press any keys to continue:');             

if __name__ == "__main__":
	main()

