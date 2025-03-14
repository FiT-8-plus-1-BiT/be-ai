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

    session_details = get_info_by_db(recommended_sessions)

    # 응답데이터 구성
    response_data = {
        "user_id": user_id,
        "recommended_sessions": session_details
    }
    
    return JSONResponse(content=response_data)

def get_info_by_db(session_ids):
    """세션 ID 리스트를 기반으로 세션 정보를 가져오는 함수"""
    if not session_ids:
        return {}

    conn = get_db_connection()
    try:
        # 여러 세션정보를 한 번에 가져옴
        format_strings = ','.join(['%s'] * len(session_ids))
        query = f"""
            SELECT session_id, title, session_image, summary, start_time, end_time 
            FROM sessions 
            WHERE session_id IN ({format_strings})
        """
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(query, tuple(session_ids))
            session_info = cursor.fetchall()
    finally:
        conn.close()

    # session_id를 키로 하고 나머지 정보를 값으로 하는 딕셔너리 생성
    return {session["session_id"]: {
        "title": session["title"],
        "session_image": session["session_image"],
        "summary": session["summary"],
        "start_time": session["start_time"].strftime("%Y-%m-%d %H:%M:%S") if session["start_time"] else None,
        "end_time": session["end_time"].strftime("%Y-%m-%d %H:%M:%S") if session["end_time"] else None
    } for session in session_info}