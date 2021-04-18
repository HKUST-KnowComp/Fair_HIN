import json

threshold = {}
threshold['dp'] = {'dev_high':0.1084,'dev_med':0.1626,'dev_low':0.2168,\
				'test_high':0.0992,'test_med':0.1488,'test_low':0.1984}
threshold['eo'] = {'dev_high':0.0590,'dev_med':0.0884,'dev_low':0.1179,\
				'test_high':0.0524,'test_med':0.0786,'test_low':0.1047}

with open('thres_fair_conditions.json','w') as f:
	json.dump(threshold,f)

# with open('data.json', 'w') as f:
#     json.dump(data, f)

# # Reading data back
# with open('data.json', 'r') as f:
#     data = json.load(f)
