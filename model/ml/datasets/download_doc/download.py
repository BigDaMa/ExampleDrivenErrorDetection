import os

#https://1-800-medigap.com/doctors/Georgia/AUSTELL/

for i in range(10100, 11100):
    os.system("wget -O /home/felix/data_more/doctors/" + str(i) + ".html https://www.medicompare.care/news/get_hospital_doctors/" + str(i))