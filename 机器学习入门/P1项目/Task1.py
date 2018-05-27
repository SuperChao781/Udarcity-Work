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
任务1：
短信和通话记录中一共有多少电话号码？每个号码只统计一次。
输出信息：
"There are <count> different telephone numbers in the records."
"""

phonenum_count = 0;   
phonenum_list = [];
            
                

for index in range(len(texts)):
    phone_num = texts[index][0];
    if phone_num not in phonenum_list:
       phonenum_count += 1;
       phonenum_list.append(phone_num);                    
    phone_num = texts[index][1];
    if phone_num not in phonenum_list:
       phonenum_count += 1;
       phonenum_list.append(phone_num);                    

for index in range(len(calls)):
    phone_num = calls[index][0];
    if phone_num not in phonenum_list:
       phonenum_count += 1;
       phonenum_list.append(phone_num);                    
    phone_num = calls[index][1];
    if phone_num not in phonenum_list:
       phonenum_count += 1;
       phonenum_list.append(phone_num);                    

outcomes = \
"There are <{0}> different telephone numbers in the records.".format(str(phonenum_count));

print(outcomes)    


