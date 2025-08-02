import time
string = input("Enter a string ")

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:',.<>?/~`  "

result= ''

for i in range(len(string)):
    for j in alphabet:
        to_print= result +j
        print(to_print)
        time.sleep(0.01)
        if j in string[i]:
            result += j
            break
        
        

        
        