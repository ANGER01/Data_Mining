
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
            del dic[key]
                    
    return dic

k = 9
test = "lkasjdfaslkdjkjfbasldkjfhhlaskdjh"

stuff = misra_gries(k, test)
print(stuff)