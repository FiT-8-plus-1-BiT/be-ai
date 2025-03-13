import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
import numpy as np

def calculate_similarity(users, sessions, my_sessions, tags):
    users['combined_text'] = users['job'].fillna('') + ' ' + users['years'].fillna(0).astype(str) + ' ' + users['interests'].fillna('')
    
    vectorizer = TfidfVectorizer()
    text_features = vectorizer.fit_transform(users['combined_text'])
    
    scaler = MinMaxScaler()
    years_scaled = scaler.fit_transform(users[['years']].fillna(0))
    
    user_features = hstack([text_features, years_scaled])
    user_similarity_df = pd.DataFrame(cosine_similarity(user_features), index=users['user_id'], columns=users['user_id'])

    # 컨텐츠 기반 필터링
    # 세션별 유사도 계산
    tags_grouped = tags.groupby("session_id").agg({
        "field": lambda x: " ".join(x.dropna().unique()),
        "topic": lambda x: " ".join(x.dropna().unique()),
        "type": lambda x: " ".join(x.dropna().unique()),
        "level": lambda x: " ".join(x.dropna().unique())
    }).reset_index()

    # 레벨을 숫자로 매핑
    level_mapping = {'B': 1, 'I': 2, 'A': 3} # Beginner, Intermediate, Advanced -> 수정가능
    tags_grouped['level_numeric'] = tags_grouped['level'].map(level_mapping).fillna(0)

    # 태그 생성
    session_tags = pd.merge(sessions, tags_grouped, on="session_id", how="left")
    session_tags['level_weight'] = session_tags['level_numeric'] * 2    
    session_tags['combined_text'] = session_tags['field'] + " " + session_tags['topic'] + " " + session_tags['type'] + " " + (session_tags['level_weight'] * 3).astype(str)

    tfidf_features = vectorizer.fit_transform(session_tags['combined_text'].fillna(""))
    session_similarity_df = pd.DataFrame(cosine_similarity(tfidf_features), index=session_tags['session_id'], columns=session_tags['session_id'])

    user_item_matrix = my_sessions.pivot(index="user_id", columns="session_id", values="session_id").notnull().astype(int)

    # my_sessions에 없는 경우에도 매트릭스는 0으로 채움
    all_users = users['user_id'].unique()
    user_item_matrix = user_item_matrix.reindex(all_users, fill_value=0)

    return user_similarity_df, session_similarity_df, user_item_matrix

