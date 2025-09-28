from data_checker import check_data_quality
data  =  [1, 2, float('nan'), 'hello', None]
result = check_data_quality(data)
print(result)