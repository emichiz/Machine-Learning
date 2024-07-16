import asyncio
import time
import types
import numpy as np
import websockets
import json
import sqlite3
import pyodbc
import pandas as pd
import joblib
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.encoding import OneHotEncoder
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import pytz
import tkinter as tk
from tkinter import messagebox
from playsound import playsound
import webbrowser



def ajustar_valor_intervalo(valor, limite_inferior, limite_superior):
    if valor is not None:
        if valor < limite_inferior:
            valor = limite_inferior
        elif valor > limite_superior:
            valor = limite_superior
    return valor


import csv
# Configurações de conexão com o servidor SQL
server = 'xxxxxx'  # Seu servidor SQL
database = 'xxxxxx'  # Seu banco de dados
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

# Conectando ao servidor SQL
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

column_names = [
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
    "token_liquidity_x",
    "token_market_cap",
    "token_created_since_horas",
    "token_created_since_min",
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
    "token_price_change_m5",
    "token_created_dia",
    "token_created_data",
    "token_created_hora",
    "data_hora"
]

# Criando um DataFrame vazio com os nomes das colunas
df = pd.DataFrame(columns=column_names)
# Lista para armazenar os valores de cada par

def calcular_venda(data_hora, previsao):
    # Converter a string de data_hora para um objeto datetime
    data_hora_obj = datetime.strptime(data_hora, "%H:%M")

    # Adicionar a previsão à data_hora
    data_hora_venda = data_hora_obj + timedelta(minutes=previsao)

    # Retornar a venda no formato HH:MM
    return data_hora_venda.strftime("%H:%M")


url = "wss://io.dexscreener.com/dex/screener/pairs/h24/1?rankBy[key]=trendingScoreH6&rankBy[order]=desc&filters[chainIds][0]=bsc&filters[chainIds][1]=arbitrum&filters[chainIds][2]=base&filters[chainIds][3]=avalanche&filters[chainIds][4]=degenchain&filters[liquidity][min]=100&filters[pairAge][max]=1&filters[buys][h1][min]=10&filters[sells][h1][min]=5"







async def main(url, data):
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

    def obter_horario_brasil():
        # Defina o fuso horário para o Brasil
        brasil_timezone = pytz.timezone('America/Sao_Paulo')

        # Obtém a hora atual no fuso horário do Brasil
        hora_atual_brasil = datetime.now(brasil_timezone)
        hora_atual_brasil = hora_atual_brasil.strftime("%H:%M")



        return hora_atual_brasil

    def ajustar_valor_intervalo(valor, limite_inferior, limite_superior):


        if valor is not None:
            if valor < limite_inferior:
                valor = limite_inferior
            elif valor > limite_superior:
                valor = limite_superior
        return valor


    try:

        async with websockets.connect(url, extra_headers=headers, timeout=150) as websocket:

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

                        token_liquidity_x = pair["liquidity"]["usd"] if "liquidity" in pair else None

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

                        data_hora = obter_horario_brasil()

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
                            float(token_liquidity_x) if token_liquidity_x is not None else None,
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
                            token_created_hora,
                            data_hora
                        ]

                        # Adicionando os valores do par ao DataFrame
                        data.append(VALUES)













                    except (KeyError, AssertionError):
                         pass


    except asyncio.exceptions.TimeoutError:
        pass


if __name__ == '__main__':
    while True:
        data = []
        asyncio.run(main(url, data))
        df = pd.DataFrame(data, columns=column_names)
        df['token_img_key'] = df['token_img_key'].apply(lambda x: '1' if x != '0' else x)


        # Primeira alteração: substituir valores na coluna 'dex_id' com mais de 20 caracteres por "outros"
        df.loc[df['dex_id'].str.len() > 20, 'dex_id'] = 'outros'

        # Terceira alteração: substituir valores NULL na coluna 'token_h1_to_m5_buys' por 0
        df['token_h1_to_m5_buys'].fillna(0, inplace=True)

        df['token_created_data'] = pd.to_datetime(df['token_created_data'], format='%Y%m%d')

        df = df[(df['token_created_since_min'] >= 10) & (df['token_created_since_min'] <= 30)]



        #cat e num vars
        cat_vars = ['chain_id', 'dex_id', 'token_created_dia', 'token_website', 'token_twitter', 'token_eti',
                    'token_header', 'token_img_key', 'token_created_hora']

        num_vars = ['token_m5_buys','token_m5_sells', 'token_h1_buys', 'token_h1_sells', 'token_liquidity_x', 'token_market_cap', 'token_links', 'token_price_change_h1', 'token_price_change_m5', 'token_h1_to_m5_buys']




        X = df[cat_vars + num_vars]

        X[cat_vars] = X[cat_vars].astype('category')




        limiar = 0.50
        # Carregando o modelo de um arquivo
        modelo120 = joblib.load('modelo120')
        proba120 = modelo120.predict_proba(X)[:, 1]
        prev120 = (proba120 >= limiar).astype(int)

        modelo90 = joblib.load('modelo90')
        proba90 = modelo90.predict_proba(X)[:, 1]
        prev90 = (proba90 >= limiar).astype(int)

        modelo60= joblib.load('modelo60')
        proba60 = modelo60.predict_proba(X)[:, 1]
        prev60 = (proba60 >= limiar).astype(int)

        df_novo = X.copy()
        colunas_token = ['token_address', 'token_name', 'token_symbol', 'token_created_data', 'token_created_hora', 'data_hora' ]
        df_novo[colunas_token] = df[colunas_token]
        df_novo['proba120'] = proba120
        df_novo['prev120'] = prev120
        df_novo['proba90'] = proba90
        df_novo['prev90'] = prev90
        df_novo['proba60'] = proba60
        df_novo['prev60'] = prev60
        df_novo['previsao'] = np.where(df_novo['prev120'] == 1, 120,
                                       np.where(df_novo['prev90'] == 1, 90,
                                                np.where(df_novo['prev60'] == 1, 60, 0)))

        df_novo['probabilidade'] = df_novo[['proba120', 'proba90', 'proba60']].max(axis=1)
        # Função para calcular a venda



        df_novo['venda'] = df_novo.apply(lambda row: calcular_venda(row['data_hora'], row['previsao']), axis=1)
        df_novo = df_novo[(df_novo['previsao'] > 0) ]

        df_m = df_novo[['chain_id', 'token_address', 'token_name', 'venda', 'probabilidade', 'previsao']].copy()

        #inner
        df_excel = pd.read_excel("df_excel.xlsx")

        # Concatenar os DataFrames existentes e os novos DataFrames
        df_combined = pd.concat([df_excel, df_m], ignore_index=True)

        linhas_unicas = df_combined.drop_duplicates(subset=['token_address', 'previsao'], keep=False)
        linhas_comuns = df_m.merge(linhas_unicas, on=['token_address', 'chain_id', 'token_name', 'venda', 'probabilidade', 'previsao' ], how='inner')

        df_combined.to_excel("df_excel.xlsx", index=False)

        df_novo = linhas_comuns[['chain_id', 'token_address', 'token_name', 'venda', 'probabilidade']].copy()

        hora_atual = datetime.now(pytz.timezone('America/Sao_Paulo'))


        print("Hora:")
        print(hora_atual)
        print(df_novo)

        if not df_novo.empty:

            # Emitir um som

            html_table = df_novo.to_html(index=False)
            with open('dataframe.html', 'w') as f:
                f.write(html_table)
            webbrowser.open('dataframe.html')
            playsound('seu_caminho')








        df_m.to_html('df_m.html', index=False)
        print("ciclo finalizado")




        time.sleep(300)

