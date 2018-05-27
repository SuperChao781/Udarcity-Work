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
任务4:
电话公司希望辨认出可能正在用于进行电话推销的电话号码。
找出所有可能的电话推销员:
这样的电话总是向其他人拨出电话，
但从来不发短信、接收短信或是收到来电


请输出如下内容
"These numbers could be telemarketers: "
<list of numbers>
电话号码不能重复，每行打印一条，按字典顺序排序后输出。
"""

'''
step 1
建立4个表：
拨打电话表
接听电话表
发短信电话表
收短信电话表
'''
call_from_list = [];
call_to_list = [];
text_from_list = [];
text_to_list = [];

for call in calls:
    call_from_list.append(call[0]);
    call_to_list.append(call[1]);                     

for text in texts:
    text_from_list.append(text[0]);
    text_to_list.append(text[1]);                     

#删除重复号码
call_from_list = list(set(call_from_list))
call_to_list = list(set(call_to_list))
text_from_list = list(set(text_from_list))
text_to_list = list(set(text_to_list))

'''
step2
任何一个电话如果在第一个表里，而不在其它任何一个表内，则满足要求
'''
telemarketers = [];
for phonenum in call_from_list:
    if(phonenum not in call_to_list and phonenum not in text_from_list and phonenum not in text_to_list):
        telemarketers.append(phonenum);             

#列表排序
telemarketers = sorted(telemarketers);

      
#输出结果
print("These numbers could be telemarketers: ");
for phonenum in telemarketers:
    print(phonenum);

