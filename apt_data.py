import requests
from pymongo import MongoClient
import xml.etree.ElementTree as ET
from datetime import datetime
import time

def get_apt_trade_data(year, month):
    url = "http://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
    params = {
        'serviceKey': 'yhCRir3QVIa+nL/vjH0uhPiZh9d8qxs3NTzqH1c4XCORTz2sawR52fE+M3/o8ze1qgXQq3dKApGO+8j35QdVNw==',
        'LAWD_CD': '11110',
        'DEAL_YMD': f"{year}{month:02d}"
    }
    
    try:
        response = requests.get(url, params=params)
        root = ET.fromstring(response.content)
        
        # 응답 결과 코드 확인
        result_code = root.find('.//resultCode').text
        if result_code != '000':
            print(f"Error response code: {result_code}")
            return []

        items = []
        for item in root.findall('.//item'):
            data = {
                'aptDong': item.find('aptDong').text,
                'aptNm': item.find('aptNm').text,
                'buildYear': int(item.find('buildYear').text) if item.find('buildYear').text else None,
                'buyerGbn': item.find('buyerGbn').text,
                'cdealDay': item.find('cdealDay').text,
                'cdealType': item.find('cdealType').text,
                'dealAmount': int(item.find('dealAmount').text.replace(',', '')) if item.find('dealAmount').text else None,
                'dealDay': int(item.find('dealDay').text) if item.find('dealDay').text else None,
                'dealMonth': int(item.find('dealMonth').text) if item.find('dealMonth').text else None,
                'dealYear': int(item.find('dealYear').text) if item.find('dealYear').text else None,
                'dealingGbn': item.find('dealingGbn').text,
                'estateAgentSggNm': item.find('estateAgentSggNm').text,
                'excluUseAr': float(item.find('excluUseAr').text) if item.find('excluUseAr').text else None,
                'floor': int(item.find('floor').text) if item.find('floor').text else None,
                'jibun': item.find('jibun').text,
                'landLeaseholdGbn': item.find('landLeaseholdGbn').text,
                'rgstDate': item.find('rgstDate').text,
                'sggCd': item.find('sggCd').text,
                'slerGbn': item.find('slerGbn').text,
                'umdNm': item.find('umdNm').text,
                # 메타 데이터 추가
                'created_at': datetime.now(),
                'data_year': year,
                'data_month': month
            }
            
            # 빈 문자열을 None으로 변환
            for key, value in data.items():
                if value == ' ' or value == '':
                    data[key] = None
                    
            items.append(data)
            
        return items
    except Exception as e:
        print(f"Error fetching data for {year}/{month}: {str(e)}")
        return []

def save_to_mongodb(data):
    try:
        client = MongoClient("mongodb+srv://takeelf:dnjsgh11!!aA@real-estate-cluster.sx265.mongodb.net/?retryWrites=true&w=majority&appName=real-estate-cluster")
        db = client['real_estate']
        collection = db['transaction_price']
        
        if data:
            result = collection.insert_many(data)
            print(f"Successfully inserted {len(result.inserted_ids)} records for {data[0]['data_year']}/{data[0]['data_month']:02d}")
        
    except Exception as e:
        print(f"MongoDB Error: {str(e)}")
    finally:
        client.close()

def main():
    for year in range(2006, 2025):
        for month in range(1, 13):
            # 미래 데이터 제외
            if year == 2024 and month > 2:
                continue
                
            print(f"Fetching data for {year}/{month:02d}")
            data = get_apt_trade_data(year, month)
            
            if data:
                save_to_mongodb(data)
            
            # API 호출 제한 고려
            time.sleep(1)

if __name__ == "__main__":
    main()
