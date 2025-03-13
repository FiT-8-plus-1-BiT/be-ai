from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pymysql
import threading
import time
import logging
import pandas as pd

from model.CalculateSimilarity import calculate_similarity
from model.HybridRecommendSessions import recommend_sessions_hybrid

app = FastAPI()

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='fit',
        password='fit1234!',
        db='fit',
        ssl={'ssl': {}}
    )

def load_data():
    conn = get_db_connection()
    
    users = pd.read_sql_query("SELECT * FROM users", conn)
    sessions = pd.read_sql_query("SELECT * FROM sessions", conn)
    my_sessions = pd.read_sql_query("SELECT * FROM my_sessions", conn)
    tags = pd.read_sql_query("SELECT * FROM tags", conn)

    conn.close()
    return users, sessions, my_sessions, tags

def update_data():
    global users, sessions, my_sessions, tags
    global user_similarity_df, session_similarity_df, user_item_matrix
    
    while True:
        try:
            users, sessions, my_sessions, tags = load_data()
            user_similarity_df, session_similarity_df, user_item_matrix = calculate_similarity(users, sessions, my_sessions, tags)
            logging.info("데이터 갱신 완료")
        except Exception as e:
            logging.error("데이터 갱신 중 오류 발생")
        time.sleep(600)  # 10분마다 실행

# 서버 시작시 자동으로 데이터갱신 스레드 실행
thread = threading.Thread(target=update_data, daemon=True)
thread.start()

@app.get("/recommend")
async def recommend(user_id: int = Query(..., description="사용자 ID")):
    users, sessions, my_sessions, tags = load_data()
    
    user_similarity_df, session_similarity_df, user_item_matrix = calculate_similarity(users, sessions, my_sessions, tags)

    recommended_sessions = recommend_sessions_hybrid(user_id, user_similarity_df, session_similarity_df, user_item_matrix, top_n=5)

    return JSONResponse(content={"user_id": user_id, "recommended_sessions": recommended_sessions})
