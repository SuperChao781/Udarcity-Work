import time
import pandas as pd
import numpy as np

CITY_DATA = { 'chicago': 'chicago.csv',
              'new york city': 'new_york_city.csv',
              'washington': 'washington.csv' }

city_list = ['chicago','new york city','washington'];
month_list = ['all','january','february','march','april','may','june'];
day_list = ['all','monday','tuesday','wednesday','thursday','friday','saturday','sunday'];

           
def Get_Items(inputList,Output,errorOutput):    
    while(True):
        inputString = input(Output + str(inputList));   
        inputString = inputString.lower();   #输入取小写
        if inputString in inputList:
            return inputString;
        else:
            print(errorOutput);
    
           
#接受输入 城市-月-日 并检查输入的正确性 输入错误时允许重复输入       
def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    """
    print('Hello! Let\'s explore some US bikeshare data!')
    # TO DO: get user input for city (chicago, new york city, washington). HINT: Use a while loop to handle invalid inputs
    city = Get_Items(city_list,'Please input the city name,you can choose it from:','Error!Please input the correct city name.');
    month = Get_Items(month_list,'Please input the month name,you can choose it from：','Error!Please input the correct month name.');
    day = Get_Items(day_list,'Please input the day name,you can choose it from：','Error!Please input the correct day name.');
    
    print('-'*40)
    return city, month, day

#导入数据
def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if applicable.

    Args:
        (str) city - name of the city to analyze
        (str) month - name of the month to filter by, or "all" to apply no month filter
        (str) day - name of the day of week to filter by, or "all" to apply no day filter
    Returns:
        df - Pandas DataFrame containing city data filtered by month and day
    """
    # load data file into a dataframe
    df = pd.read_csv(CITY_DATA[city]);

    # convert the Start Time column to datetime
    df['Start Time'] = df['Start Time'] = pd.to_datetime(df['Start Time']);

    # extract month and day of week from Start Time to create new columns
    df['month'] = df['Start Time'].dt.month;
    df['day_of_week'] = df['Start Time'].dt.weekday;


    # month_num by month if applicable
    month_num = month_list.index(month);#all-0 January-1 February-2 and so on
    if month_num != 0:   #which means month != 'all'
        # use the index of the months list to get the corresponding int 
        # filter by month to create the new dataframe
        df = df[df['month']==month_num];

    # filter by day of week if applicable
##匹配星期的方法一
    day_num = day_list.index(day);#all-0 Monday-1 Tuesday-2 and so on
    if day_num != 0:
        # filter by day of week to create the new dataframe
        day_num = day_num - 1;#Monday-0 Tuesday-1 .. Saturday-5 Sunday-6
        
        df = df[df['day_of_week']==day_num];
    return df

#获取最常见的月 日 起始时间
def time_stats(df):
    """Displays statistics on the most frequent times of travel."""

    print('\nCalculating The Most Frequent Times of Travel...\n')
    start_time = time.time()

    # TO DO: display the most common month 取出最常见的月
    month_most_popular = df['month'].mode()[0];
    month_most_popular_num = month_list[month_most_popular];
    print('The Month that people travel most is:'+month_most_popular_num);
        
    # TO DO: display the most common day of week 取出最常见的天
    day_most_popular = df['day_of_week'].mode()[0];
    day_most_popular_num = day_list[day_most_popular+1];
    print('The Day that people travel most is:'+day_most_popular_num);


    # TO DO: display the most common start hour 取出最常见的起始时间
    df['start hour'] = df['Start Time'].dt.hour;
    hour_most_popular = df['start hour'].mode()[0];
    print('The Hour that people start for travel most is:'+str(hour_most_popular));
    
    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

#获取最常见的起始、结束站点
def station_stats(df):
    """Displays statistics on the most popular stations and trip."""

    print('\nCalculating The Most Popular Stations and Trip...\n')
    start_time = time.time()

    # TO DO: display most commonly used start station
    month_most_start_station = df['Start Station'].mode()[0];
    print('The Start Station that people travel most is:'+month_most_start_station);

    # TO DO: display most commonly used end station
    month_most_end_station = df['End Station'].mode()[0];
    print('The End Station that people travel most is:'+month_most_end_station);


    # TO DO: display most frequent combination of start station and end station trip
    df['Start-End Station']=df['Start Station']+'-'+df['End Station'];  #生成新的一列，该列为起始站-终点站组合
    month_most_start_end_station = df['Start-End Station'].mode()[0].split('-');
    print('The Start-End Station that people travel most is:{0} to {1}.'.format(month_most_start_end_station[0],month_most_start_end_station[1]));

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def trip_duration_stats(df):
    """Displays statistics on the total and average trip duration."""

    print('\nCalculating Trip Duration...\n')
    start_time = time.time()

    # TO DO: display total travel time
    totalTime = df['Trip Duration'].sum();
    print('The total time that people cost for travel most is:' + str(totalTime) + ' seconds.');

    # TO DO: display mean travel time
    averageTime = totalTime/len(df);
    print('The average time that people cost for each travel most is:' + str(averageTime) + ' seconds.');

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)


def user_stats(df):
    """Displays statistics on bikeshare users."""

    print('\nCalculating User Stats...\n')
    start_time = time.time()

    # TO DO: Display counts of user types
    User_Type_count = df['User Type'].value_counts();
    for i in range(len(User_Type_count)):
        print('The count of user type'+ User_Type_count.index[i] + ' is:' + str(User_Type_count[i]));

    if('Gender' not in df.columns):#数据中不包含性别，则不执行下述操作
        return;

    # TO DO: Display counts of gender
    User_Type_count = df['Gender'].value_counts();
    for i in range(len(User_Type_count)):
        print('The count of '+User_Type_count.index[i] + ' is: ' + str(User_Type_count[i]));

    # TO DO: Display earliest, most recent, and most common year of birth
    min_age = df['Birth Year'].min();
    max_age = df['Birth Year'].max();
    common_age = df['Birth Year'].mode()[0];
    print('The earliest birth year is:'+str(min_age));
    print('The most recent birth year is:'+str(max_age));
    print('The most common birth year is:'+str(common_age));
    

    print("\nThis took %s seconds." % (time.time() - start_time))
    print('-'*40)

df = pd.DataFrame();
def main():
    #while True:
        city, month, day = get_filters();#读取用户输入
        
        global df
        df = load_data(city, month, day);#根据用户的选择，读取数据

        print('our city:'+city);
        print('our month:'+month);
        print('our day:'+day);
        
        
        
        
        time_stats(df)
        station_stats(df)
        trip_duration_stats(df)
        user_stats(df)

        #restart = input('\nWould you like to restart? Enter yes or no.\n')
        #if restart.lower() != 'yes':
            #break


if __name__ == "__main__":
	main()
