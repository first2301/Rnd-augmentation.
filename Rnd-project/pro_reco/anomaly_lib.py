from pycaret.anomaly import AnomalyExperiment
# import ray

# @ray.remote
def train_caret_anomaly(updated_df):
    s = AnomalyExperiment()
    s.setup(data=updated_df, session_id=123, normalize_method = 'minmax')
    abod = s.create_model('abod')
    cluster = s.create_model('cluster')
    #cof = s.create_model('cof')
    iforest = s.create_model('iforest')
    histogram = s.create_model('histogram')
    knn = s.create_model('knn')
    lof = s.create_model('lof')
    #svm = s.create_model('svm')
    pca = s.create_model('pca')
    mcd = s.create_model('mcd')
    #sod = s.create_model('sod')
    # sos = s.create_model('sos')

    abod_anomalies = s.assign_model(abod)
    cluster_anomalies = s.assign_model(cluster)
    #cof_anomalies = s.assign_model(cof)
    iforest_anomalies = s.assign_model(iforest)
    histogram_anomalies = s.assign_model(histogram)
    knn_anomalies = s.assign_model(knn)
    lof_anomalies = s.assign_model(lof)
    #svm_anomalies = s.assign_model(svm)
    pca_anomalies = s.assign_model(pca)
    mcd_anomalies = s.assign_model(mcd)
    #sod_anomalies = s.assign_model(sod)
    # sos_anomalies = s.assign_model(sos)
    
    return {'abod_anomalies': abod_anomalies, 'cluster_anomalies': cluster_anomalies, 'iforest_anomalies': iforest_anomalies,
            'histogram_anomalies': histogram_anomalies, 'knn_anomalies': knn_anomalies, 'lof_anomalies': lof_anomalies,
            'pca_anomalies':pca_anomalies, 'mcd_anomalies': mcd_anomalies}

# from pycaret.anomaly import AnomalyExperiment

# def train_caret_anomaly(updated_df):
#     s = AnomalyExperiment()
#     s.setup(data=updated_df, session_id=123, normalize_method = 'minmax')
#     abod = s.create_model('abod')
#     cluster = s.create_model('cluster')
#     iforest = s.create_model('iforest')
#     histogram = s.create_model('histogram')
#     knn = s.create_model('knn')
#     lof = s.create_model('lof')
#     pca = s.create_model('pca')
#     mcd = s.create_model('mcd')

#     abod_anomalies = s.assign_model(abod)
#     cluster_anomalies = s.assign_model(cluster)
#     iforest_anomalies = s.assign_model(iforest)
#     histogram_anomalies = s.assign_model(histogram)
#     knn_anomalies = s.assign_model(knn)
#     lof_anomalies = s.assign_model(lof)
#     pca_anomalies = s.assign_model(pca)
#     mcd_anomalies = s.assign_model(mcd)
#     return {'abod_anomalies': abod_anomalies, 'cluster_anomalies': cluster_anomalies, 'iforest_anomalies': iforest_anomalies,
#             'histogram_anomalies': histogram_anomalies, 'knn_anomalies': knn_anomalies, 'lof_anomalies': lof_anomalies,
#             'pca_anomalies':pca_anomalies, 'mcd_anomalies': mcd_anomalies}




# def compare_anomalies(X):
#     # 빈 DataFrame 생성
#     models = {'Iforest': IsolationForest(), 'DBSCAN':DBSCAN(), 'KMeans':KMeans(), 'GaussianMixture': GaussianMixture()}
#     model_score_df = pd.DataFrame()
#     for name, model in models.items():
#         model_fit = make_pipeline(StandardScaler(), model).fit(X)
#         if name == 'Iforest':  # For Isolation Forest, we need anomaly scores instead of predictions
#             y_scores = -model_fit.score_samples(X)
#             y_pred = model_fit.predict(X)  # Assigning y_pred to prevent UnboundLocalError
#         else:
#             y_pred = model_fit.predict(X)
#             y_scores = -model_fit.decision_function(X)  # Use decision function for other clustering models

#         model_score_df.at[name, 'Silhouette Score'] = silhouette_score(X, y_pred)
#         model_score_df.at[name, 'Davies-Bouldin Score'] = davies_bouldin_score(X, y_pred)
#         model_score_df.at[name, 'Anomaly Score'] = y_scores.mean()

#     return model_score_df.T

