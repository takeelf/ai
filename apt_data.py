import requests
from pymongo import MongoClient
import xml.etree.ElementTree as ET
from datetime import datetime
import time

code_list = ['11000', '11110', '11140', '11170', '11200', '11215', '11230', '11260', '11290', '11305', '11320', '11350', '11380', '11410', '11440', '11470', '11500', '11530', '11545', '11560', '11590', '11620', '11650', '11680', '11710', '11740', '21000', '21110', '21140', '21170', '21200', '21230', '21260', '21290', '21320', '21350', '21380', '21410', '21440', '22000', '22110', '22140', '22170', '22200', '22230', '22260', '22290', '23000', '23110', '23140', '23170', '23200', '23230', '23260', '23290', '23320', '24000', '24110', '24140', '24170', '24200', '24230', '24260', '25000', '25110', '25140', '25170', '25200', '25230', '25260', '25290', '25320', '25350', '25380', '25410', '25440', '25470', '25500', '25530', '25710', '25720', '25730', '25740', '25750', '25760', '25770', '25780', '25790', '25800', '25810', '25820', '25830', '25840', '25850', '25860', '25870', '25880', '25890', '25900', '25910', '25920', '25930', '25940', '25950', '25960', '25970', '25980', '25990', '26000', '26110', '26140', '26170', '26200', '26230', '26260', '26290', '26320', '26350', '26380', '26410', '26440', '26470', '26500', '26530', '26710', '26720', '26730', '26740', '26750', '26760', '26770', '26780', '26790', '26800', '26810', '26820', '26830', '26840', '26850', '26860', '26870', '26880', '26890', '26900', '26910', '26920', '26930', '26940', '26950', '26960', '26970', '26980', '26990', '27000', '27110', '27140', '27170', '27200', '27230', '27260', '27290', '27710', '27720', '27730', '27740', '27750', '27760', '27770', '27780', '27790', '27800', '27810', '27820', '27830', '27840', '27850', '27860', '27870', '27880', '27890', '27900', '27910', '27920', '27930', '27940', '27950', '27960', '27970', '27980', '27990', '28000', '28110', '28140', '28170', '28200', '28230', '28260', '28710', '28720', '28730', '28740', '28750', '28760', '28770', '28780', '28790', '28800', '28810', '28820', '28830', '28840', '28850', '28860', '28870', '28880', '28890', '28900', '28910', '28920', '28930', '28940', '28950', '28960', '28970', '28980', '28990', '29000', '29110', '29140', '29170', '29200', '29710', '29720', '29730', '29740', '29750', '29760', '29770', '29780', '29790', '29800', '29810', '29820', '29830', '29840', '29850', '29860', '29870', '29880', '29890', '29900', '29910', '29920', '29930', '29940', '29950', '29960', '29970', '29980', '29990', '30000', '30110', '30140', '30170', '30200', '30230', '31000', '31110', '31140', '31170', '31200', '31230', '31260', '31290', '31320', '31350', '31380', '31710', '31720', '31730', '31740', '31750', '31760', '31770', '31780', '31790', '31800', '31810', '31820', '31830', '31840', '31850', '31860', '31870', '31880', '31890', '31900', '31910', '31920', '31930', '31940', '31950', '31960', '31970', '31980', '31990', '32000', '32110', '32130', '32710', '32720', '32730', '32740', '32750', '32760', '32770', '32780', '32790', '32800', '32810', '32820', '32830', '32840', '32850', '32860', '32870', '32880', '32890', '32900', '32910', '32920', '32930', '32940', '32950', '32960', '32970', '32980', '32990', '33000', '33110', '33130', '33710', '33720', '33730', '33740', '33750', '33760', '33770', '33780', '33790', '33800', '33810', '33820', '33830', '33840', '33850', '33860', '33870', '33880', '33890', '33900', '33910', '33920', '33930', '33940', '33950', '33960', '33970', '33980', '33990', '34000', '34110', '34130', '34310', '34330', '34350', '34360', '34370', '34380', '34390', '34400', '34410', '34420', '34430', '34440', '34450', '34460', '34470', '34480', '34490', '34500', '34510', '34520', '34530', '34540', '34550', '34560', '34570', '34580', '34590', '34600', '34610', '34620', '34630', '34640', '34650', '34660', '34670', '34680', '34690', '34700', '34710', '34720', '34730', '34740', '34750', '34760', '34770', '34780', '34790', '34800', '34810', '34820', '34830', '34840', '34850', '34860', '34870', '34880', '34890', '34900', '34910', '34920', '34930', '34940', '34950', '34960', '34970', '34980', '34990', '35000', '35110', '35130', '35310', '35320', '35330', '35340', '35350', '35360', '35370', '35380', '35390', '35400', '35410', '35420', '35430', '35440', '35450', '35460', '35470', '35480', '35490', '35500', '35510', '35520', '35530', '35540', '35550', '35560', '35570', '35580', '35590', '35600', '35610', '35620', '35630', '35640', '35650', '35660', '35670', '35680', '35690', '35700', '35710', '35720', '35730', '35740', '35750', '35760', '35770', '35780', '35790', '35800', '35810', '35820', '35830', '35840', '35850', '35860', '35870', '35880', '35890', '35900', '35910', '35920', '35930', '35940', '35950', '35960', '35970', '35980', '35990', '36000', '36110', '36130', '36310', '36320', '36330', '36340', '36350', '36360', '36370', '36380', '36390', '36400', '36410', '36420', '36430', '36440', '36450', '36460', '36470', '36480', '36490', '36500', '36510', '36520', '36530', '36540', '36550', '36560', '36570', '36580', '36590', '36600', '36610', '36620', '36630', '36640', '36650', '36660', '36670', '36680', '36690', '36700', '36710', '36720', '36730', '36740', '36750', '36760', '36770', '36780', '36790', '36800', '36810', '36820', '36830', '36840', '36850', '36860', '36870', '36880', '36890', '36900', '36910', '36920', '36930', '36940', '36950', '36960', '36970', '36980', '36990', '37000', '37110', '37130', '37310', '37320', '37330', '37340', '37350', '37360', '37370', '37380', '37390', '37400', '37410', '37420', '37430', '37440', '37450', '37460', '37470', '37480', '37490', '37500', '37510', '37520', '37530', '37540', '37550', '37560', '37570', '37580', '37590', '37600', '37610', '37620', '37630', '37640', '37650', '37660', '37670', '37680', '37690', '37700', '37710', '37720', '37730', '37740', '37750', '37760', '37770', '37780', '37790', '37800', '37810', '37820', '37830', '37840', '37850', '37860', '37870', '37880', '37890', '37900', '37910', '37920', '37930', '37940', '37950', '37960', '37970', '37980', '37990', '38000', '38110', '38130', '38310', '38320', '38330', '38340', '38350', '38360', '38370', '38380', '38390', '38400', '38410', '38420', '38430', '38440', '38450', '38460', '38470', '38480', '38490', '38500', '38510', '38520', '38530', '38540', '38550', '38560', '38570', '38580', '38590', '38600', '38610', '38620', '38630', '38640', '38650', '38660', '38670', '38680', '38690', '38700', '38710', '38720', '38730', '38740', '38750', '38760', '38770', '38780', '38790', '38800', '38810', '38820', '38830', '38840', '38850', '38860', '38870', '38880', '38890', '38900', '38910', '38920', '38930', '38940', '38950', '38960', '38970', '38980', '38990', '39000', '39110', '39130', '39310', '39320', '39330', '39340', '39350', '39360', '39370', '39380', '39390', '39400', '39410', '39420', '39430', '39440', '39450', '39460', '39470', '39480', '39490', '39500', '39510', '39520', '39530', '39540', '39550', '39560', '39570', '39580', '39590', '39600', '39610', '39620', '39630', '39640', '39650', '39660', '39670', '39680', '39690', '39700', '39710', '39720', '39730', '39740', '39750', '39760', '39770', '39780', '39790', '39800', '39810', '39820', '39830', '39840', '39850', '39860', '39870', '39880', '39890', '39900', '39910', '39920', '39930', '39940', '39950', '39960', '39970', '39980', '39990', '41000', '41110', '41111', '41113', '41115', '41117', '41130', '41131', '41133', '41135', '41150', '41170', '41171', '41173', '41190', '41210', '41220', '41250', '41270', '41271', '41273', '41280', '41281', '41285', '41287', '41290', '41310', '41360', '41370', '41390', '41410', '41430', '41450', '41460', '41461', '41463', '41465', '41480', '41500', '41550', '41570', '41590', '41610', '41630', '41650', '41670', '41800', '41820', '41830', '42000', '42110', '42130', '42150', '42170', '42190', '42210', '42230', '42720', '42730', '42750', '42760', '42770', '42780', '42790', '42800', '42810', '42820', '42830', '43000', '43110', '43111', '43112', '43113', '43114', '43130', '43150', '43720', '43730', '43740', '43745', '43750', '43760', '43770', '43800', '44000', '44110', '44130', '44131', '44133', '44150', '44170', '44180', '44200', '44210', '44230', '44250', '44270', '44710', '44720', '44730', '44740', '44750', '44760', '44770', '44780', '44790', '44800', '44810', '44820', '44830', '44840', '44850', '44860', '44870', '44880', '44890', '44900', '44910', '44920', '44930', '44940', '44950', '44960', '44970', '44980', '44990', '45000', '45110', '45111', '45113', '45130', '45140', '45180', '45190', '45210', '45710', '45720', '45730', '45740', '45750', '45770', '45790', '45800', '46000', '46110', '46130', '46150', '46170', '46230', '46710', '46720', '46730', '46770', '46780', '46790', '46800', '46810', '46820', '46830', '46840', '46860', '46870', '46880', '46890', '46900', '46910', '47000', '47110', '47111', '47113', '47130', '47150', '47170', '47190', '47210', '47230', '47250', '47280', '47290', '47720', '47730', '47750', '47760', '47770', '47780', '47790', '47800', '47810', '47820', '47830', '47840', '47850', '47900', '47920', '47930', '47940', '48000', '48110', '48120', '48121', '48123', '48125', '48127', '48129', '48170', '48220', '48240', '48250', '48270', '48310', '48330', '48710', '48720', '48730', '48740', '48750', '48760', '48770', '48775', '48780', '48790', '48795', '48800', '48810', '48820', '48830', '48840', '48850', '48860', '48870', '48880', '48890', '49000', '49110', '49130', '49710', '49720', '50000', '50110', '50130', '51000', '51110', '51130', '51150', '51170', '51190', '51210', '51230', '51720', '51730', '51750', '51760', '51770', '51780', '51790', '51800', '51810', '51820', '51830', '52000', '52110', '52111', '52113', '52130', '52140', '52180', '52190', '52210', '52710', '52720', '52730', '52740', '52750', '52770', '52790', '52800']

def process_and_save_data(year, month):
    url = "http://apis.data.go.kr/1613000/RTMSDataSvcAptTrade/getRTMSDataSvcAptTrade"
    total_saved = 0
    
    # MongoDB 연결 (한 번만 연결)
    try:
        client = MongoClient("mongodb+srv://takeelf:dnjsgh11!!aA@real-estate-cluster.sx265.mongodb.net/?retryWrites=true&w=majority&appName=real-estate-cluster")
        db = client['real_estate']
        collection = db['transaction_price']
        
        for code in code_list:
            params = {
                'serviceKey': 'yhCRir3QVIa+nL/vjH0uhPiZh9d8qxs3NTzqH1c4XCORTz2sawR52fE+M3/o8ze1qgXQq3dKApGO+8j35QdVNw==',
                'LAWD_CD': code,
                'DEAL_YMD': f"{year}{month:02d}"
            }
            
            try:
                print(f"Fetching data for region code {code}, {year}/{month:02d}")
                response = requests.get(url, params=params)
                root = ET.fromstring(response.content)
                
                # 응답 결과 코드 확인
                result_code = root.find('.//resultCode').text
                if result_code != '000':
                    print(f"Error response code for region {code}: {result_code}")
                    continue

                # 한 지역의 데이터를 담을 리스트
                items_to_save = []
                
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
                        'sggCd': code,
                        'slerGbn': item.find('slerGbn').text,
                        'umdNm': item.find('umdNm').text,
                        'created_at': datetime.now(),
                        'data_year': year,
                        'data_month': month
                    }
                    
                    # 빈 문자열을 None으로 변환
                    for key, value in data.items():
                        if value == ' ' or value == '':
                            data[key] = None
                            
                    items_to_save.append(data)
                
                # 한 지역의 데이터를 바로 MongoDB에 저장
                if items_to_save:
                    result = collection.insert_many(items_to_save)
                    total_saved += len(result.inserted_ids)
                    print(f"Saved {len(result.inserted_ids)} records for region {code}")
                
                # API 호출 제한을 위한 딜레이
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing region {code}: {str(e)}")
                continue
        
        print(f"\nTotal records saved for {year}/{month:02d}: {total_saved}")
        
    except Exception as e:
        print(f"MongoDB Error: {str(e)}")
    finally:
        client.close()

def main():
    for year in range(2010, 2025):
        for month in range(1, 13):
            # 미래 데이터 제외
            if year == 2024 and month > 2:
                continue
                
            print(f"\nProcessing {year}/{month:02d}")
            process_and_save_data(year, month)
            
            # 월별 처리 후 잠시 대기
            time.sleep(1)

if __name__ == "__main__":
    main()



