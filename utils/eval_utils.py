"""Stable majority voting for ARC prediction candidates."""
import json

def get_majority_vote(predictions):
    vote_count = {}
    list_map = {}
    for i in range(len(predictions)):
        label = json.dumps(predictions[i])
        list_map[label] = predictions[i]
        if label not in vote_count:
            vote_count[label] = 0
        vote_count[label] += 1
    sorted_votes = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)
    sorted_lists = [
        {"prediction": list_map[item[0]], "votes": item[1]} for item in sorted_votes
    ]
    return sorted_lists
