import re
import jieba

text1 = "JGood_is_a_handsome@boy,#he is_cool,&clever, and so on...你吃苹果了吗"
# print(re.sub(r'\_', '   ', text1, 5, flags=0))

text1 = ' '.join(jieba.cut(text1))
print(text1)

# filters = ['#', '!', '@', '$', '_']
# text2 = '|'.join(filters)
# print(text2)
# text3 = re.sub(text2, '++', text1)
# print(text3)
