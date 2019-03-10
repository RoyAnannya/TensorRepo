#python log preprocessor

import csv
import re

print("build files")
# 1,3,4,6--->train
# 2,5,8---->test
with open('data/log_file.csv', 'a',True,'utf-8',newline='') as out_file:
    with open('C:\\Glassbox\\Logs\\glassboxlog.log') as in_file:

        writer = csv.writer(out_file)
        writer.writerow(['date','time', 'label', 'title','content'])

        for line in in_file:
            line = re.sub(" +"," ",line)
            columns = line[:-1].split(' ')
            if (columns[2]==('INFO')):
                columns[2]='1'
            elif (columns[2]==('DEBUG')):
                columns[2]='2'
            elif (columns[2]==('ERROR')):
                columns[2]='3'
            else :
                columns[2]='0'
            # print ("====>>",type(columns[2]))   
            columns[4] = ' '.join(columns[4:])
            # columns[4] = '\''+(columns[4])
            columns[4] = (columns[4])[2:]
            # print (columns[4])
            writer.writerow(columns[:5])