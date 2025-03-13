import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)  # 기본 로그 레벨 설정

def recommend_sessions_hybrid(user_id, user_similarity_df, session_similarity_df, user_item_matrix, top_n=5):
    """
    하이브리드 추천 시스템 (사용자 기반 CF + 아이템 기반 CF + 행동 기반 추천)
    :param user_id: 추천 대상 사용자 ID
    :param user_similarity_df: 사용자 기반 협업 필터링 유사도 행렬
    :param session_similarity_df: 세션 기반 협업 필터링 유사도 행렬
    :param user_item_matrix: 사용자-세션 매트릭스
    :param top_n: 추천할 세션 개수
    :return: 추천된 세션 리스트
    """
    if user_id not in user_item_matrix.index:
        print(f"사용자 {user_id}가 `user_item_matrix`에 없습니다. 빈 데이터로 초기화합니다.")
        new_user_row = pd.DataFrame(0, index=[user_id], columns=user_item_matrix.columns)
        user_item_matrix = pd.concat([user_item_matrix, new_user_row])
    
    logging.debug(f"user_similarity_df 데이터 확인 : \n{user_similarity_df.loc[user_id].sort_values(ascending=False).head(5)}")
    logging.debug(f"session_similarity_df 데이터 확인 : \n{session_similarity_df.loc[user_id].sort_values(ascending=False).head(11)}")

    # 사용자가 등록한 세션
    user_sessions = user_item_matrix.loc[user_id] 
    liked_sessions = user_sessions[user_sessions > 0].index.tolist()

    # 사용자 기반 협업 필터링 (비슷한 유저들의 세션 가중치를 조정)
    similar_users = user_similarity_df.loc[user_id].drop(user_id)
    top_similar_users = similar_users.sort_values(ascending=False).head(5)

    weighted_user_sessions = (user_item_matrix.loc[top_similar_users.index].T * top_similar_users.values).T
    similar_users_sessions = weighted_user_sessions.mean()

    # 아이템 기반 협업 필터링
    similar_sessions = session_similarity_df.loc[liked_sessions].sum().sort_values(ascending=False)

    # 행동 기반 추천 (내가 좋아한 세션을 좋아한 유저들이 좋아한 세션 추천)
    users_liked_my_sessions = user_item_matrix[user_item_matrix[liked_sessions].sum(axis=1) > 0]

    user_sim_scores = user_similarity_df.loc[user_id, users_liked_my_sessions.index]
    weighted_sessions = (user_item_matrix.loc[users_liked_my_sessions.index].T * user_sim_scores.values).T
    expanded_sessions = weighted_sessions.mean()

    # 최종 추천 가중치 조정
    final_recommendations = ((0.5 * similar_users_sessions) + (0.1 * similar_sessions) + (0.4 * expanded_sessions)).sort_values(ascending=False)

    recommended_sessions = final_recommendations.drop(liked_sessions, errors='ignore')


    return list(recommended_sessions.head(top_n).index)
