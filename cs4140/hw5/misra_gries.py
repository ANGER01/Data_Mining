import math

def misra_gries(k, m):
    dic = {}
    for value in m:
        if value in dic:
            dic[value] = dic[value]+1
        
        elif len(dic) < k-1:
            dic[value] = 1
        
        else:
            to_remove = []
            for key in dic.keys():
                dic[key] = dic[key] - 1
                if dic[key] == 0:
                    to_remove.append(key)
            for val in to_remove:
                del dic[val]
                    
    return dic

with open("S1_hw5.csv", "r", encoding="utf-8") as file:
    content = file.read()  # Read the entire file as a string

with open("S2_hw5.csv", "r", encoding="utf-8") as file1:
    content1 = file1.read()
    
s1 = 3_000_000
s2 = 4_000_000

report1 = misra_gries(10, content)
print(report1)

total = 0
ahh = 0
for val in report1:
    total += report1[val]
    temp = ((report1[val]/s1)*100)
    ahh += temp
    print(val, temp)

print(ahh)
print("Percent off S1", ((s1 - total)/s1))
print("Error bound is 10%")
report2 = misra_gries(10, content1)
print(report2)

total1 = 0
tot_per = 0
for val in report2:
    total1 += report2[val]
    percent = ((report2[val]/s2)*100)
    tot_per += percent
    print(val, percent)

print(tot_per)    
print("Percent off S2", ((s2 - total1)/s2))
print("Error bound is 10%")


