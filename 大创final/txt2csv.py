import csv

csvFile = open("data.csv", 'w', newline='', encoding='utf-8')  # 固定格式
writer = csv.writer(csvFile)  # 固定格式
csvRow = []  # 用来存储csv文件中一行的数据

f = open("household_power_consumption.txt", 'r', encoding='GB2312')

for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)

f.close()

csvFile.close()
