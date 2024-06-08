import numpy as np
from .utils.metrics import metricor


def generate_curve(label,score,slidingWindow):
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d


def get_metrics(score, labels, metric='vus', slidingWindow=None):
    metrics = {}
    if metric == 'vus':
        grader = metricor()
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, 2*slidingWindow)
        
        metrics['R_AUC_ROC'] = R_AUC_ROC
        metrics['R_AUC_PR'] = R_AUC_PR
        metrics['VUS_ROC'] = VUS_ROC
        metrics['VUS_PR'] = VUS_PR

        return metrics
    
    elif metric == 'all' or metric != 'vus':
        grader = metricor()
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, 2*slidingWindow)

        
        from .basic_metrics import basic_metricor
        
        grader = basic_metricor()
        AUC_ROC, Precision, Recall, F, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, plot_ROC=False)
        _, _, AUC_PR = grader.metric_PR(labels, score)


        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        discrete_score = np.array(score > 0.5, dtype=np.float32)
        events_pred = convert_vector_to_events(discrete_score)
        events_gt = convert_vector_to_events(labels)
        Trange = (0, len(discrete_score))
        affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)

        metrics['AUC_ROC'] = AUC_ROC
        metrics['AUC_PR'] = AUC_PR
        metrics['Precision'] = Precision
        metrics['Recall'] = Recall
        metrics['F'] = F
        metrics['Precision_at_k'] = Precision_at_k
        metrics['Rprecision'] = Rprecision
        metrics['Rrecall'] = Rrecall
        metrics['RF'] = RF
        metrics['R_AUC_ROC'] = R_AUC_ROC
        metrics['R_AUC_PR'] = R_AUC_PR
        metrics['VUS_ROC'] = VUS_ROC
        metrics['VUS_PR'] = VUS_PR
        metrics['Affiliation_Precision'] = affiliation_metrics['Affiliation_Precision']
        metrics['Affiliation_Recall'] = affiliation_metrics['Affiliation_Recall']

        if metric == 'all':
            return metrics
        else:
            return metrics[metric]
