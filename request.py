import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Material_Quantity':5, 'Additive_Catalyst':7, 'Ash_Component':10,
                            'Water_Mix':2, 'Plasticizer':9, 'Moderate_Aggregator':6,
                            'Refined_Aggregator':2, 'Formulation_Duration':9})

print(r.json())