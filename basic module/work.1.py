# # # -*- coding: utf-8 -*-
# l=[]
# for num in range(2020, 2520):
#     if num %7 == 0 and num%5 !=  0:
#         l.append(num)
# print (l)
        
        
        
# "第一题"

# l=[]
# a=[1,2,3,4]
# n=0
# for i in a:
#     for j in a:
#         for k in a:
#             if len(set((i,j,k))) == 3 :
#                 num=i*100+j*10+k
#                 l.append(num)
#                 n=n+1
#                 print (n)
# print(l)

# "第二题"

# n=eval(input('a num:'))
# a_dict=dict()
# for i in range(1, 3):
#         a=str(n)
#         b=n*n
#         b=str(b)
#         a_dict.update({a:b})
#         n=n-1
# print(a_dict)

# "第三题"

# name=list(input('name:').split())
# NAME= [a.upper()for a in name]
# NAME.sort()
# print(NAME)

# "第四题"

# word=WORD=num=0
# a=list(input('a sentence:'))
# for i in a:
#     if i.isdigit():
#         num=num+1
#     elif i.isupper():
#         WORD+=1
#     elif i.islower():
#         word+=1
# print(num,',',word,',',WORD)

# "第五题"

# a=list(input('money:').split())
# b_dict=dict(D=0,W=0)
# b_dict['D']=b_dict['D']+int(a[1])
# b_dict['W']=b_dict['W']+int(a[3])
# print('money=',b_dict['D']-b_dict['W'])

# "第六题"
# def check(str):
#     if not str.isdigit() and not str.isupper() and not str.islower():
#           print('wrong')
#           return 0
#     else:
#         return 1
    
# string='@#$'
# a = list(input('enter your password:').split(','))
# for i in range(0,len(a)):
#     if len(a[i])>6 and len(a[i])<12 and check(a[i]) and string in a[i]:
#                 print(a[i])
        


# "第七题"

# for y in range(1, 35):
#     x=35-y
#     if 2*x+4*y==94:
#       print ('chick\'number is',str(x),'rabbit\'s number is',str(y))

# "第八题"

class Person(object):
    def __get__(self):        
        pass
class Male(Person):
      def __init__(self,gender):
          Person.__get__(self)
          self.gender = gender
          print(self.gender)
a=Male('male')

"第九题"



class Person(object):
    def __init__(self):      
        pass     
class Chinese(Person):
    def __init__(self, language): 
        Person.__init__(self) 
        self.language = language
        print(  self.language)
c = Chinese( 'Chinese')

              


        
