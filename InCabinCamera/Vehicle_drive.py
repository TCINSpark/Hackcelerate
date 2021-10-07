# with open('/Users/dasaradibudhi/Downloads/heartrate.rtf', 'r') as file:
#     text = file.read()
#     print(text)

# with open("/Users/dasaradibudhi/Downloads/heartrate.rtf") as infile:
#     counter=0
#     for line in infile:
#         counter +=1 
#         print(line)
#         print(counter)


from striprtf.striprtf import rtf_to_text
heartrate_array = []

with open('/Users/dasaradibudhi/OneDrive - TOYOTA Connected India Pvt. Ltd/Project/TCDS/Hackathon/Hackcelerate/Code/hr.txt', 'r') as file:
    counter = 0 
    for line in file:
        counter += 1
        # txt = rtf_to_text(line.rstrip())
        # txt = txt.replace('''\\''',"")
        print(counter)
        print(int(line))
        heartrate_array.append(line.split())

print(heartrate_array.pop(10))