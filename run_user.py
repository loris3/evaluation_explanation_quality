
def run_user(idx, user, url, df_user_study):
    n_learn = 16
    n_eval = 16
    n_users = 10

    mu_got_it_right_pre=0.5
    sigma_got_it_right_pre=0.05
    mu_gain = 0.2
    sigma_gain = 0.1

    def guess(detector_label,p):
        return detector_label if bool(np.random.choice([0,1],p=[1-p, p])) else not detector_label

    import requests
    import json
    import numpy as np
    import pandas as pd
    user_dist_without = lambda : np.clip(np.random.normal(mu_got_it_right_pre, sigma_got_it_right_pre, 1)[0], 0,1)
    user_dist_gain = lambda : np.clip(np.random.normal(mu_gain, sigma_gain, 1)[0], -1,1)
  
    res = requests.get(url+"/auth/"+ user["access_token"])

    print(res.text)
    auth_token = json.loads(res.text)
    headers = {'Content-Type': 'application/json','Authorization': "Bearer "+auth_token, "Content-Type": "application/json",}

    requests.post(url+"/api/submitParticipantInfo", json={
    "has_seen_explanation_methods_before": "yes",
    "has_seen_OTHERS_before": "yes",
    "level_of_expertise": "is-researcher-explainability",
    "familiarity_with_chatgpt": "occasional-use",
    "prefers_monochromatic_methods": "yes" if user["access_token"] == "DDEBUG" else "no"
    }, headers=headers)
    # go to phase 2
    requests.post(url+"/api/completeCurrentPhase", json={"expected": 0}, headers=headers)
    requests.post(url+"/api/completeCurrentPhase", json={"expected": 1}, headers=headers)
    requests.post(url+"/api/completeCurrentPhase", json={"expected": 2}, headers=headers)

    res = requests.get(url+"/api/state", headers=headers)
    state = json.loads(res.text)

    # user_df = pd.read_sql_query("SELECT * FROM users", connection) # update as group is assigned now
    # user = user_df.iloc[idx]
   # print(user[["detector", "explainer"]])
    # return state
    df_user_documents = df_user_study.loc[df_user_study.groupby("Detector").groups[state["detector"]],:].reset_index(drop=True)
    for doc_nr, row in df_user_documents.iterrows():
        p_without = user_dist_without()
        requests.post(url+"/api/submitPhase2", json={"ID": doc_nr, "label": guess(row["f(b)"], p_without)}, headers=headers)
    requests.post(url+"/api/completeCurrentPhase", json={"expected": 3}, headers=headers)

    for doc_nr, row in df_user_documents.iterrows():
        json_ = {"lickert-q{}-{}".format(question_nr, doc_nr): str(np.random.choice([1,2,3,4,5], p=[0.1,0.2,0.1,0.4,0.2])) for question_nr in range(1,4)}
        json_["document_nr"] = doc_nr
        requests.post(url+"/api/submitPhase3", json=json_, headers=headers)

    requests.post(url+"/api/completeCurrentPhase", json={"expected": 4}, headers=headers)
    for doc_nr, row in df_user_documents.iterrows():
        p_with = np.clip(p_without + user_dist_gain(), 0,1)
        requests.post(url+"/api/submitPhase4", json={"ID": doc_nr, "label": guess(row["f(b)"], p_with)}, headers=headers)
    requests.post(url+"/api/completeCurrentPhase", json={"expected": 5}, headers=headers)
