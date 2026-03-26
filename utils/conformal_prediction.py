import numpy as np

options = ["A", "B", "C", "D", "E", "F"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def LAC_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    """
    Apply conformal prediction using LAC score function.
    """
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            pred_sets_all[m+"_"+fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m+"_"+fs]["cal"]
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"])
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_scores.append(1 - probs[options.index(truth_answer)])
            
            n = len(cal_logits_data)
            q_level = np.ceil((n+1) * (1-alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method='higher')
            
            pred_sets = {}
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"])
                ps = []
                for ii, p in enumerate(probs):
                    if p >= 1 - qhat:
                        ps.append(options[ii])
                if len(ps) == 0:
                    ps.append(options[np.argmax(probs)])
                pred_sets[str(row["id"])] = ps
            pred_sets_all[m+"_"+fs] = pred_sets
    return pred_sets_all

def APS_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    """
    Apply conformal prediction using APS score function.
    """
    ada_pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            ada_pred_sets_all[m+"_"+fs] = {}
            cal_scores = []
            cal_logits_data = logits_data_all[m+"_"+fs]["cal"]
            for idx, row in enumerate(cal_logits_data):
                probs = softmax(row["logits_options"])
                truth_answer = cal_raw_data[idx]["answer"]
                assert cal_raw_data[idx]["id"] == row["id"]
                cal_pi = np.argsort(probs)[::-1]
                cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                cal_sum_r = np.take_along_axis(cal_sum, cal_pi.argsort(), axis=0)
                cal_score = cal_sum_r[options.index(truth_answer)]
                cal_scores.append(cal_score)
            
            n = len(cal_logits_data)
            q_level = np.ceil((n+1) * (1-alpha)) / n
            qhat = np.quantile(cal_scores, q_level, method='higher')
            
            pred_sets = {}
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            for idx, row in enumerate(test_logits_data):
                probs = softmax(row["logits_options"])
                cal_pi = np.argsort(probs)[::-1]
                cal_sum = np.take_along_axis(probs, cal_pi, axis=0).cumsum()
                ps = []
                ii = 0
                while ii < len(cal_sum) and cal_sum[ii] <= qhat:
                    op_id = cal_pi[ii]
                    ps.append(options[op_id])
                    ii += 1
                if len(ps) == 0:
                    op_id = cal_pi[ii]
                    ps.append(options[op_id])
                pred_sets[str(row["id"])] = ps
            ada_pred_sets_all[m+"_"+fs] = pred_sets
    return ada_pred_sets_all