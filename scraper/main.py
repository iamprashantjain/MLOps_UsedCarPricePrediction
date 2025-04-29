import concurrent.futures
import requests
import json
import pandas as pd
import os

headers = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-US,en;q=0.9',
    'clientid': '186638435.1745859574',
    'content-type': 'application/json',
    'origin': 'https://www.cars24.com',
    'priority': 'u=1, i',
    'referer': 'https://www.cars24.com/',
    'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'cross-site',
    'source': 'WebApp',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
    'x_experiment_id': 'c8f4ecad-7a16-44b2-bedd-ec74e6cbd227',
    'x_tenant_id': 'INDIA_CAR_LISTING',
    'x_user_city_id': '1',
}

city_ids = list(range(1,100001))
file_name = 'cars24_allCityIds_data.csv'

def extract_all_cityId_data(city_id):    
    print(city_id)
    
    json_data = {
        'searchFilter': [],
        'cityId': str(city_id),
        'sort': 'bestmatch',
        'size': 10000,
        'filterVersion': 1,
    }

    try:
        response = requests.post('https://car-catalog-gateway-in.c24.tech/listing/v1/buy-used-car', headers=headers, json=json_data)        
        if response.status_code == 200:
            json_string = response.content.decode('utf-8')
            data_dict = json.loads(json_string)
            df = pd.DataFrame(data_dict['content'])
            try:
                df = df[['appointmentId', 'maskedRegNum', 'cityId', 'listingPrice', 'carName', 'make', 'model', 'variant', 'year', 'transmissionType', 'bodyType', 'fuelType', 'ownership', 'registrationDate', 'cityRto', 'color', 'odometer', 'emiDetails', 'modelGroup', 'oemServiceHistoryAvailable', 'fitnessAge', 'features']]
                file_exists = os.path.exists(file_name)
                df.to_csv(file_name, mode='a', header=not file_exists, index=False)
            except Exception as e:
                pass

    except Exception as e:
        print(e)


def run_concurrent_requests():
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        executor.map(extract_all_cityId_data, city_ids)


if __name__ == "__main__":
    run_concurrent_requests()