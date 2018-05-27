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
任务2: 哪个电话号码的通话总时间最长? 不要忘记，用于接听电话的时间也是通话时间的一部分。
输出信息:
"<telephone number> spent the longest time, <total time> seconds, on the phone during
September 2016.".

提示: 建立一个字典，并以电话号码为键，通话总时长为值。
这有利于你编写一个以键值对为输入，并修改字典的函数。
如果键已经存在于字典内，为键所对应的值加上对应数值；
如果键不存在于字典内，将此键加入字典，并将它的值设为给定值。
"""

call_dict = {};
call_dict_max = {'':0};
max_time = 0;
#建立字典  统计所有电话的通话总时间            
for index in range(len(calls)):
    #拨打电话时间
    if(calls[index][0] in call_dict.keys()):
        call_dict[calls[index][0]] += int(calls[index][3]);
    else:
        call_dict[calls[index][0]] = int(calls[index][3]);         
    #接听电话时间
    if(calls[index][1] in call_dict.keys()):
        call_dict[calls[index][1]] += int(calls[index][3]);
    else:
        call_dict[calls[index][1]] = int(calls[index][3]); 
#寻找并记录通话时间最长的号码有对应时间 
for key,value in call_dict.items():
    if(value > max_time):
        call_dict_max = {key:value};
        max_time = value;

for key,value in call_dict_max.items():
    outcome = "<{0}> spent the longest time, \
<{1}> seconds, on the phone during September 2016.".format(key,value);
    print(outcome);
        