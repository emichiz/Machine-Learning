import asyncio
import time
import types

import websockets
import json
import sqlite3
import pyodbc

from datetime import datetime


def ajustar_valor_intervalo(valor, limite_inferior, limite_superior):
    if valor is not None:
        if valor < limite_inferior:
            valor = limite_inferior
        elif valor > limite_superior:
            valor = limite_superior
    return valor



# Configurações de conexão com o servidor SQL
server = 'xxxxx'  # Seu servidor SQL
database = 'xxxxx'  # Seu banco de dados
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# Conectando ao servidor SQL
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

DATA = []
FIELDNAMES = [
  "chain_id",
  "dex_id",
  "pair_address",
  "token_address",
  "token_name",
  "token_symbol",
  "token_m5_buys",
  "token_m5_sells",
  "token_h1_buys",
  "token_h1_sells",
  "token_h1_to_m5_buys",
  "token_liquidity",
  "token_market_cap",
  "token_created_at",
  "token_created_since",
  "token_eti",
  "token_header",
  "token_website",
  "token_twitter",
  "token_links",
  "token_img_key",
  "token_price_usd",
  "token_price_change_h24",
  "token_price_change_h6",
  "token_price_change_h1",
  "token_price_change_m5"
]
urls = []

for i in range(1, 10):
    
    url = f"wss://io.dexscreener.com/dex/screener/pairs/h24/{i}?rankBy[key]=trendingScoreH6&rankBy[order]=desc&filters[liquidity][min]=100&filters[pairAge][max]=6"
    
    urls.append(url)


async def main():

    tasks = []
    for url in urls:
        tasks.append(extract_data_from_url(url))

    await asyncio.gather(*tasks)



async def extract_data_from_url(url):
    headers = {
      "Host": "io.dexscreener.com",
      "Connection": "Upgrade",
      "Pragma": "no-cache",
      "Cache-Control": "no-cache",
      "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
      "Upgrade": "websocket",
      "Origin": "https://dexscreener.com",
      "Sec-WebSocket-Version": 13,
      "Accept-Encoding": "gzip, deflate, br, zstd",
      "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    def ajustar_valor_intervalo(valor, limite_inferior, limite_superior):
        if valor is not None:
            if valor < limite_inferior:
                valor = limite_inferior
            elif valor > limite_superior:
                valor = limite_superior
        return valor


    try:

        async with websockets.connect(url, extra_headers=headers, timeout=60) as websocket:

                message_raw = await websocket.recv()



                message = json.loads(message_raw)


                pairs = message.get("pairs", [])

                for pair in pairs:
                    try:
                        chain_id = pair["chainId"]
                        dex_id = pair["dexId"]
                        pair_address = pair.get("pairAddress")

                        assert pair_address, "Pair address is missing"

                        token_address = pair["baseToken"]["address"]
                        token_name = pair["baseToken"]["name"] [:999]
                        token_symbol = pair["baseToken"]["symbol"] [:49]

                        token_txns = pair.get("txns", {})

                        token_m5_buys = token_txns.get("m5", {}).get("buys", 0)
                        token_m5_sells = token_txns.get("m5", {}).get("sells", 0)

                        token_h1_buys = token_txns.get("h1", {}).get("buys", 0)
                        token_h1_sells = token_txns.get("h1", {}).get("sells", 0)

                        token_h1_to_m5_buys = round(token_m5_buys * 12 / token_h1_buys, 2) if token_m5_buys else None

                        token_liquidity = pair["liquidity"]["usd"] if "liquidity" in pair else None

                        token_market_cap = pair["marketCap"] if "marketCap" in pair else None
                        token_created_at_raw = pair['pairCreatedAt'] if "pairCreatedAt" in pair else None
                        token_created_at2 = token_created_at_raw / 1000 if "pairCreatedAt" in pair else None
                        token_created_at2 = datetime.utcfromtimestamp(token_created_at2) if "pairCreatedAt" in pair else None
                        now_utc = datetime.utcnow() if "pairCreatedAt" in pair else None
                        token_created_since_min = round((now_utc - token_created_at2).total_seconds() / 60,
                                                    2) if "pairCreatedAt" in pair else None
                        token_created_since_horas = token_created_since_min / 60 if "pairCreatedAt" in pair else None

                        token_eti = pair.get("ear", False)
                        token_header = pair.get("profile", {}).get("header", False)
                        token_website = pair.get("profile", {}).get("website", False)
                        token_twitter = pair.get("profile", {}).get("twitter", False)
                        token_links = pair.get("profile", {}).get("linkCount", False)
                        token_img_key = pair.get("profile", {}).get("imgKey", False)

                        token_price_usd = pair["priceUsd"]
                        token_price_change_h24 = float(pair["priceChange"]["h24"])
                        token_price_change_h6 = float(pair["priceChange"]["h6"])
                        token_price_change_h1 = float(pair["priceChange"]["h1"])
                        token_price_change_m5 = float(pair["priceChange"]["m5"])

                        #Novos values
                        token_price_change_h24 = ajustar_valor_intervalo(token_price_change_h24, -9999999999,
                                                                         9999999999)
                        token_price_change_h6 = ajustar_valor_intervalo(token_price_change_h6, -9999999999, 9999999999)
                        token_price_change_h1 = ajustar_valor_intervalo(token_price_change_h1, -9999999999, 9999999999)
                        token_price_change_m5 = ajustar_valor_intervalo(token_price_change_m5, -9999999999, 9999999999)



                        #dia da semana
                        data_hora_atual = datetime.now()
                        dia_semana_numero = data_hora_atual.weekday()
                        dias_semana = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira',
                                       'sábado', 'domingo']

                        token_created_dia = dias_semana[dia_semana_numero]
                        # Obtendo apenas a data
                        token_created_data = int(data_hora_atual.date().strftime("%Y%m%d")) if data_hora_atual else None


                        # Obtendo apenas a hora
                        token_created_hora = int(data_hora_atual.strftime(":%H").replace(":", "")) if data_hora_atual else None

                        # Função para ajustar valores dentro de um intervalo específico
                        def ajustar_valor_intervalo(valor, limite_inferior, limite_superior):
                            if valor is not None:
                                if valor < limite_inferior:
                                    valor = limite_inferior
                                elif valor > limite_superior:
                                    valor = limite_superior
                            return valor




                        VALUES = [
                            (chain_id),
                            (dex_id),
                            pair_address,
                            token_address,
                            token_name,
                            token_symbol,
                            int(token_m5_buys),
                            int(token_m5_sells),
                            int(token_h1_buys),
                            int(token_h1_sells),
                            float(token_h1_to_m5_buys) if token_h1_to_m5_buys is not None else None,
                            float(token_liquidity) if token_liquidity is not None else None,
                            float(token_market_cap) if token_market_cap is not None else None,
                            token_created_since_horas,
                            float(token_created_since_min) if token_created_since_min is not None else None,
                            int(token_eti),
                            int(token_header),
                            token_website,
                            token_twitter,
                            int(token_links),
                            token_img_key,
                            float(token_price_usd),
                            token_price_change_h24,
                            token_price_change_h6,
                            token_price_change_h1,
                            token_price_change_m5,
                            token_created_dia,
                            token_created_data,
                            token_created_hora
                        ]


                        cursor.execute('''
                        INSERT INTO
                        DEX( chain_id, dex_id, pair_address, token_address, token_name, token_symbol, token_m5_buys, token_m5_sells, token_h1_buys, token_h1_sells, token_h1_to_m5_buys, token_liquidity, token_market_cap, token_created_since_horas, token_created_since_min, token_eti, token_header, token_website, token_twitter, token_links, token_img_key, token_price_usd,  token_price_change_h24, token_price_change_h6, token_price_change_h1, token_price_change_m5, token_created_dia, token_created_data, token_created_hora)
                        VALUES(?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (chain_id, dex_id, pair_address, token_address, token_name, token_symbol, token_m5_buys, token_m5_sells, token_h1_buys, token_h1_sells, token_h1_to_m5_buys, token_liquidity, token_market_cap, token_created_since_horas, token_created_since_min, token_eti, token_header, token_website, token_twitter, token_links, token_img_key, token_price_usd,  token_price_change_h24, token_price_change_h6, token_price_change_h1, token_price_change_m5, token_created_dia, token_created_data, token_created_hora ))


                        

                        



                    except (KeyError, AssertionError):
                         pass

                        

    except asyncio.exceptions.TimeoutError:
        pass


if __name__ == '__main__':
    while True:
        asyncio.run(main())
        conn.commit()
        print("ciclo finalizado")
        time.sleep(480)
        
