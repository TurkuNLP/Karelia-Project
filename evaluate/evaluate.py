import argparse
import json
import unicodedata
import Levenshtein



def normalize(entity):
    # lowercase, remove punctuation, remove whitespace

    entity = "".join([c for c in entity.strip().lower().split()]) # lowercase, remove spacing
    chars = []
    for c in entity: # remove punctuation
        if unicodedata.category(c).startswith("P"):
            continue
        chars.append(c)
    return "".join(chars)

    
def fuzzy_match(gold, ann, threshold=0.75):
    # all pairs
    scores = []
    for g_e in gold:
        for a_e in ann:
            score = max(Levenshtein.ratio(g_e, a_e), Levenshtein.ratio(a_e, g_e))
            scores.append((score, g_e, a_e))
    sorted_scores = sorted(scores, key=lambda x: x[0])
    final_pairs = []
    for (score, g, a) in sorted_scores:
        if score > threshold:
            if g in gold and a in ann:
                final_pairs.append((g,a,score))
                gold.remove(g)
                ann.remove(a)
    return final_pairs, gold, ann

def micro_f(gold, pred, args):
    
    # How many of the empty gold fields are empty also in prediction
    empty_gold = 0 # how many of the gold fields are empty
    empty_correct = 0 # how many of those are correctly annotated

    # Micro F-score
    false_positives = 0 # prediction not in gold
    false_negatives = 0 # gold entity not predicted
    true_positives = 0 # prediction in gold

    for gold_story, pred_story in zip(gold, pred):
    
        assert gold_story["index"] == pred_story["index"]
        assert gold_story["primary_person_id"] == pred_story["primary_person_id"]
        
        field_names = ["person_hobbies", "person_social_orgs", "spouse_hobbies", "spouse_social_orgs"]
        for field_name in field_names:

            gold_entities = set([normalize(e.strip()) for e in gold_story[field_name].split(",") if e.strip() != ""])
            pred_entities = set([normalize(e.strip()) for e in pred_story[field_name].split(",") if e.strip() != ""])
            
            
            #### entity level metrics ####
            
            # gold annotations empty (all predictions are false positives)
            if len(gold_entities) == 0:
                empty_gold += 1
                if len(pred_entities) == 0:
                    empty_correct += 1
                else:
                    false_positives += len(pred_entities)
                continue # done!
                
            # prediction empty (all gold are false negatives)
            if len(pred_entities) == 0:
                false_negatives += len(gold_entities)
                continue # done!
            
            # true positives (exact match)
            common = gold_entities & pred_entities
            true_positives += len(common)
            
            # ..what is left
            gold_entities = gold_entities - common
            pred_entities = pred_entities - common
            
            # fuzzy match
            if args.fuzzy_threshold != 1.0:
                fuzzy_pairs, gold_entities, pred_entities = fuzzy_match(gold_entities, pred_entities, threshold=args.fuzzy_threshold)
                #print("Fuzzy pairs:", len(fuzzy_pairs))
                true_positives += len(fuzzy_pairs)
            
            # fuzzy match done, the rest are false negatives or false positives
            false_positives += len(pred_entities)
            false_negatives += len(gold_entities)

            
    # print metrics
    print()
    print(f"{empty_correct} out of {empty_gold} empty gold fields correctly predicted ({(empty_correct/empty_gold)*100}%)")
    print()


    print("P/R/F")
    p = true_positives/(true_positives+false_positives) # tp/(tp+fp)
    r = true_positives/(true_positives+false_negatives) # tp/(tp+fn)
    print(f"P: {r}") 
    print(f"R: {p}") 
    print(f"F: {2*((p*r)/(p+r))}") # 2*((p*r)/(p+r))


def main(args):

    with open(args.gold, "rt", encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(args.prediction, "rt", encoding="utf-8") as f:
        prediction_data = json.load(f)

    print("Fuzzy threshold:", args.fuzzy_threshold)

    micro_f(gold_data, prediction_data, args)
    
    
if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=None, required=True, help="Gold standard annotation file")
    parser.add_argument("--prediction", default=None, required=True, help="Annotation file")
    parser.add_argument("--fuzzy-threshold", type=float, default=1.0, help="Threshold for fuzzy matches, Default: 1.0, use Exact Match [EM], which includes only very basic normalization [casing, spacing, punctuation] but not fuzzy matches")
    args = parser.parse_args()
    
    main(args)
