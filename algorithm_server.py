from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import pymysql
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import threading
import time
import numpy as np

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
            print("데이터 갱신 중")
            users, sessions, my_sessions, tags = load_data()
            user_similarity_df, session_similarity_df, user_item_matrix = calculate_similarity(users, sessions, my_sessions, tags)
            print("데이터 갱신 완료")
        except Exception as e:
            print(f"데이터 갱신 중 오류 발생: {e}")

        time.sleep(1800)  # 30분마다 실행
        
def calculate_similarity(users, sessions, my_sessions, tags):
    users['combined_text'] = users['job'].fillna('') + ' ' + users['interests'].fillna('')
    
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(users['combined_text'])
    
    scaler = MinMaxScaler()
    years_scaled = scaler.fit_transform(users[['years']].fillna(0))
    
    user_features = hstack([text_features, years_scaled])
    user_similarity_df = pd.DataFrame(cosine_similarity(user_features), index=users['user_id'], columns=users['user_id'])
    print("user_similarity_df 데이터 확인 : ", user_similarity_df.loc[13].sort_values(ascending=False).head(5))

    tags_grouped = tags.groupby("session_id").agg({
        "field": lambda x: " ".join(x.dropna().unique()),
        "topic": lambda x: " ".join(x.dropna().unique()),
        "type": lambda x: " ".join(x.dropna().unique()),
        "level": lambda x: " ".join(x.dropna().unique())
    }).reset_index()

    session_tags = pd.merge(sessions, tags_grouped, on="session_id", how="left")
    session_tags['combined_text'] = session_tags['field'] + " " + session_tags['topic'] + " " + session_tags['type'] + " " + session_tags['level']

    tfidf_features = vectorizer.fit_transform(session_tags['combined_text'].fillna(""))
    session_similarity_df = pd.DataFrame(cosine_similarity(tfidf_features), index=session_tags['session_id'], columns=session_tags['session_id'])
    print("session_similarity_df 데이터 확인 : ", session_similarity_df.head(5))  # 상위 5개 출력

    user_item_matrix = my_sessions.pivot(index="user_id", columns="session_id", values="session_id").notnull().astype(int)

    # my_sessions에 없는 경우에도 매트릭스는 0으로 채움
    all_users = users['user_id'].unique()
    user_item_matrix = user_item_matrix.reindex(all_users, fill_value=0)
    print("user_item_matrix 데이터 확인 : ", user_item_matrix.loc[13])

    return user_similarity_df, session_similarity_df, user_item_matrix


def recommend_sessions_hybrid(user_id, user_similarity_df, item_similarity_df, user_item_matrix, top_n=5):
    """
    하이브리드 추천 시스템 (사용자 기반 CF + 아이템 기반 CF + 행동 기반 추천)
    :param user_id: 추천 대상 사용자 ID
    :param user_similarity_df: 사용자 기반 협업 필터링 유사도 행렬
    :param item_similarity_df: 세션 기반 협업 필터링 유사도 행렬
    :param user_item_matrix: 사용자-세션 매트릭스
    :param top_n: 추천할 세션 개수
    :return: 추천된 세션 리스트
    """
    if user_id not in user_item_matrix.index:
        print(f"⚠️ 사용자 {user_id}가 `user_item_matrix`에 없습니다. 빈 데이터로 초기화합니다.")
        new_user_row = pd.DataFrame(0, index=[user_id], columns=user_item_matrix.columns)
        user_item_matrix = pd.concat([user_item_matrix, new_user_row])

    # 사용자 기반 협업 필터링
    similar_users = user_similarity_df.loc[user_id].drop(user_id)
    top_similar_users = similar_users.sort_values(ascending=False).head(5)
    similar_users_sessions = user_item_matrix.reindex(top_similar_users, fill_value=0)

    # 아이템 기반 협업 필터링
    user_sessions = user_item_matrix.loc[user_id]
    liked_sessions = user_sessions[user_sessions > 0].index.tolist()
    similar_sessions = item_similarity_df.loc[liked_sessions].sum().sort_values(ascending=False)
    
    # 행동 기반 추천 >> 내가 좋아한 세션을 좋아한 유저들이 좋아한 세션 추천
    users_liked_my_sessions = user_item_matrix[user_item_matrix[liked_sessions].sum(axis=1) > 0]
    expanded_sessions = users_liked_my_sessions.sum().sort_values(ascending=False)


    # 모든 추천 결과를 합쳐서 가중 평균
    final_recommendations = (
        (0.5 * similar_users_sessions.sum()) +
        (0.3 * similar_sessions) +
        (0.2 * expanded_sessions)
    ).sort_values(ascending=False)

    # 사용자가 등록한 세션 제외 추천
    recommended_sessions = final_recommendations.drop(liked_sessions, errors='ignore')

    return list(recommended_sessions.head(top_n).index)

@app.get("/recommend")
async def recommend(user_id: int = Query(..., description="사용자 ID")):
    users, sessions, my_sessions, tags = load_data()
    
    user_similarity_df, session_similarity_df, user_item_matrix = calculate_similarity(users, sessions, my_sessions, tags)

    recommended_sessions = recommend_sessions_hybrid(user_id, user_similarity_df, session_similarity_df, user_item_matrix, top_n=5)

    return JSONResponse(content={"user_id": user_id, "recommended_sessions": recommended_sessions})
