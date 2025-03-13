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
        print(f"사용자 {user_id}가 `user_item_matrix`에 없습니다. 빈 데이터로 초기화합니다.")
        new_user_row = pd.DataFrame(0, index=[user_id], columns=user_item_matrix.columns)
        user_item_matrix = pd.concat([user_item_matrix, new_user_row])
    
    # 사용자가 등록한 세션
    user_sessions = user_item_matrix.loc[user_id] 
    liked_sessions = user_sessions[user_sessions > 0].index.tolist()

    # ✅ 사용자 기반 협업 필터링 (비슷한 유저들의 세션 가중치를 조정)
    similar_users = user_similarity_df.loc[user_id].drop(user_id)
    top_similar_users = similar_users.sort_values(ascending=False).head(5)

    if not top_similar_users.empty:
        weighted_user_sessions = (user_item_matrix.loc[top_similar_users.index].T * top_similar_users.values).T
        similar_users_sessions = weighted_user_sessions.mean()  # ✅ sum() 대신 mean() 사용
    else:
        similar_users_sessions = pd.Series(0, index=user_item_matrix.columns)

    # 아이템 기반 협업 필터링 (여전히 강하지만, 조정)
    if liked_sessions:
        similar_sessions = item_similarity_df.loc[liked_sessions].sum().sort_values(ascending=False)
    else:
        similar_sessions = pd.Series(0, index=user_item_matrix.columns)

    # 행동 기반 추천 (내가 좋아한 세션을 좋아한 유저들이 좋아한 세션 추천)
    users_liked_my_sessions = user_item_matrix[user_item_matrix[liked_sessions].sum(axis=1) > 0]

    if not users_liked_my_sessions.empty:
        user_sim_scores = user_similarity_df.loc[user_id, users_liked_my_sessions.index]
        weighted_sessions = (user_item_matrix.loc[users_liked_my_sessions.index].T * user_sim_scores.values).T
        expanded_sessions = weighted_sessions.mean()  # ✅ sum() 대신 mean() 사용
    else:
        expanded_sessions = pd.Series(0, index=user_item_matrix.columns)

    # 최종 추천 가중치 조정
    final_recommendations = ((0.5 * similar_users_sessions) + (0.1 * similar_sessions) + (0.4 * expanded_sessions)).sort_values(ascending=False)

    recommended_sessions = final_recommendations.drop(liked_sessions, errors='ignore')

    similarity_results = {}

    # ✅ 추천된 세션 5개와 user_id 간의 유사도 비교
    for session_id in recommended_sessions.head(top_n + 1).index.tolist():
        similarity_results[session_id] = user_similarity_df.loc[user_id, session_id]

    print(f"\n✅ 추천된 세션과 {user_id}의 유사도:")
    for session_id, similarity in similarity_results.items():
        print(f" - 세션 {session_id} ↔ 사용자 {user_id} 유사도: {similarity}")

    # ✅ 추천된 유저 5명과 user_id 간의 유사도 비교
    top_similar_users = user_similarity_df.loc[user_id].drop(user_id).sort_values(ascending=False).head(top_n).index.tolist()

    similarity_results_users = {}
    for similar_user in top_similar_users:
        similarity_results_users[similar_user] = user_similarity_df.loc[user_id, similar_user]

    print(f"\n✅ 추천된 유저와 {user_id}의 유사도:")
    for similar_user, similarity in similarity_results_users.items():
        print(f" - 유저 {similar_user} ↔ 사용자 {user_id} 유사도: {similarity}")


    return list(recommended_sessions.head(top_n).index)
