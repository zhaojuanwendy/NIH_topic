
input ='1R01MH114888-01A1 \n  be the foundation o \n 1R01MH114888-01A1\n afafa'

first_pos = input.find('1R01MH114888-01A1')
if first_pos>=0:
    if input.find('1R01MH114888-01A1', first_pos+1)>=0:
        print(input[first_pos+len('1R01MH114888-01A1'): input.find('1R01MH114888-01A1', first_pos+1)])
    else:
        print(input[first_pos+len('1R01MH114888-01A1'):])
