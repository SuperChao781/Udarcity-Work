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
任务0:
短信记录的第一条记录是什么？通话记录最后一条记录是什么？
输出信息:
"First record of texts, <incoming number> texts <answering number> at time <time>"
"Last record of calls, <incoming number> calls <answering number> at time <time>, lasting <during> seconds"
"""

first_text = texts[0];
last_call = calls[-1];

first_text_output = "First record of texts, <{0[0]}> texts <{0[1]}>\
 at time <{0[2]}>".format(first_text);
last_call_output = "Last record of calls, <{0[0]}> calls <{0[1]}> \
 at time <{0[2]}>, lasting <{0[3]}> seconds".format(last_call);

print(first_text_output)       
print(last_call_output)         