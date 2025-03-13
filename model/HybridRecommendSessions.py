import pandas as pd

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
