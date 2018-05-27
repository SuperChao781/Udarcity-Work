"""
下面的文件将会从csv文件中读取读取短信与电话记录，
你将在以后的课程中了解更多有关读取文件的知识。
"""
import csv

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

"""
任务3:
(080)是班加罗尔的固定电话区号。
固定电话号码包含括号，
所以班加罗尔地区的电话号码的格式为(080)xxxxxxx。

第一部分: 找出被班加罗尔地区的固定电话所拨打的所有电话的区号和移动前缀（代号）。
 - 固定电话以括号内的区号开始。区号的长度不定，但总是以 0 打头。
 - 移动电话没有括号，但数字中间添加了
   一个空格，以增加可读性。一个移动电话的移动前缀指的是他的前四个
   数字，并且以7,8或9开头。
 - 电话促销员的号码没有括号或空格 , 但以140开头。

输出信息:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
代号不能重复，每行打印一条，按字典顺序排序后输出。

第二部分: 由班加罗尔固话打往班加罗尔的电话所占比例是多少？
换句话说，所有由（080）开头的号码拨出的通话中，
打往由（080）开头的号码所占的比例是多少？

输出信息:
"<percentage> percent of calls from fixed lines in Bangalore are calls
to other fixed lines in Bangalore."
注意：百分比应包含2位小数。
"""

'''
part1 program
'''
                    
def get_num_in_brackets(string):
#该函数提取一个字符串中第一个()内的内容，并返回 
#例如 输入'(080)xxxx' 返回080    
    output = '';
    for index in range(len(string)):
        character = string[index];
        if(character == '('):
            continue;
        elif(character == ')'):
            break;
        else:
            output += character;
    return output;
        
#首先提取所有主叫号码中满足格式(080)xxxxxxx的行；
BangaloreFlag = '(080)';#固定电话区号前缀
calledbyBangalore_list = [];#被班加罗尔地区固话拨打的电话的区号和移动前缀列表
calls_from_Bangalore = [call for call in calls if call[0][0:5] == BangaloreFlag];


    

#统计这些行包含的被叫号码的区号和移动前缀，存储为list；
preNum = 0;#前缀
for index in range(len(calls_from_Bangalore)):
    called = calls_from_Bangalore[index][1];

    if(called[0] == '(' and called[1] == '0'):  #固定电话
        preNum = get_num_in_brackets(called);#提取固定电话前缀
    elif(called[0:3] == '140'):#电话推销员
        preNum = '140';#提取电话推销员前缀
    elif(int(called[0]) >=7 and int(called[0]) <= 9 and called[5] == ' '):#移动电话
        preNum = called[0:4];#提取移动前缀
    else:#非法数据格式
        continue;
    if preNum not in calledbyBangalore_list:#将电话前缀添加到列表中
        calledbyBangalore_list.append(preNum);

#列表排序
calledbyBangalore_list_sorted = sorted(calledbyBangalore_list);

#输出信息
print("The numbers called by people in Bangalore have codes:");
for num in calledbyBangalore_list_sorted:
    print(num);


'''
part2 program
'''
#在上一步所有班加罗尔电话主叫的电话中，找出被叫电话也是班加罗尔电话的记录
calls_from_Bangalore_to_Bangalore = [call for call in calls_from_Bangalore if call[1][0:5] == BangaloreFlag];
#计算比例
ratio = len(calls_from_Bangalore_to_Bangalore)/len(calls_from_Bangalore)*100;                                 

outcome = "<{:.2f}> percent of calls from fixed lines in Bangalore are calls \
to other fixed lines in Bangalore.".format(ratio);
                                          
print(outcome);
















