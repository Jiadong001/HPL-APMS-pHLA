# return to parent directory
import sys 
if ".." not in sys.path:
    sys.path.append("..")

import numpy as np
import pandas as pd
import torch
import time

amino_acid_list = [ "G", "A", "V", "L", "I", "P", "F", "Y", "W", "S",
                    "T", "C", "M", "N", "Q", "D", "E", "K", "R", "H"]
hla_max_len = 182
pep_max_len = 15

def seq2token(tokenizer, hla_seq, pep_seq, hla_max_len, pep_max_len):
    pep_tokens, hla_pep_tokens = [], []
    
    assert type(hla_seq)==str
    hla_seq = hla_seq.ljust(hla_max_len, 'X')
    hla_token = tokenizer.encode(hla_seq)

    if type(pep_seq) == str:
        pep_seq = [pep_seq]
    assert type(pep_seq) == list
    for seq in pep_seq:
        seq = seq.ljust(pep_max_len, 'X')
        pep_tokens.append(tokenizer.encode(seq))    # [array]

        phla_seq = hla_seq + seq
        hla_pep_tokens.append(tokenizer.encode(phla_seq))
    
    return np.array(hla_token), np.array(pep_tokens), np.array(hla_pep_tokens)

def algorithm2a( 
    given_HLA, HLA_seq, init_peptide, 
    tokenizer, models, device,
    iteration, beam_width,
    writein_file=True, record_time=False,
    filename="supplementary_file",
    ):

    if iteration > len(init_peptide):
        iteration = len(init_peptide)

    if writein_file:
        record_file = open(f"./{filename}/al2a_{given_HLA}_len{len(init_peptide)}_iter{iteration}_model{len(models)}_{init_peptide}", "w")
        record_file.write(f">Source\n{init_peptide}")

    batch_size = 16
    source_peptides = [init_peptide]
    output_peptides = []
    mask_pos_list = []

    end_times = []
    for i in range(iteration):
        # 1. Make a pool of candidate mutated peptides
        mutant_pool = []
        for source_peptide in source_peptides:
            # (1) replace amino with <unk> and calculate saliency
            saliency_all = []
            _, _, hla_spep_token = seq2token(tokenizer, HLA_seq, source_peptide, hla_max_len, pep_max_len)
            for ind, amino in enumerate(source_peptide):
                if amino == init_peptide[ind]:              # find non-mutated position against given peptide
                    mask_token = hla_spep_token.copy()      # copy() is necessary
                    mask_token[0][hla_max_len+1+ind] = 4    # <unk> is 4
                    # print(hla_spep_token, mask_token)
                    # print(hla_spep_token.shape, mask_token.shape)

                    with torch.no_grad():
                        hla_pep_inputs = torch.LongTensor(
                            np.concatenate((hla_spep_token, mask_token), axis=0)
                            ).to(device)
                        # print(hla_pep_inputs.shape)
                        for ind, model in enumerate(models):
                            model_output = model(hla_pep_inputs)
                            model_score = model_output[:, 1] - model_output[:, 0]
                            model_score = model_score.cpu().detach().numpy()   # 1-D
                            if ind == 0:
                                y_score_all_ensemble = model_score
                            else:
                                y_score_all_ensemble = y_score_all_ensemble + model_score
                        y_score_all_ensemble = y_score_all_ensemble / len(models)
                        prob = 1 / (1+np.exp(-y_score_all_ensemble))        # sigmod, size=2
                        # print(prob)
                        saliency = prob[1] - prob[0]        # masked - origin
                        # print(saliency)
                        saliency_all.append(saliency)
                else:
                    # print(ind+1)
                    saliency_all.append(-1)

            # (2) saliency ranking
            saliency_all = np.array(saliency_all)
            mask_position = np.argsort(saliency_all)[-1]    # best position to be replaced
            # if source_peptides == [source_peptide]:         # print first round's saliency
            #     print("position:", np.argsort(saliency_all)+1)
            #     print("saliency:", np.sort(saliency_all))

            mask_pos_list.append(mask_position+1)
            # print(mask_position+1)

            # (3) replace amino at the mask_position
            for sub_amino in amino_acid_list:
                if sub_amino != source_peptide[mask_position]:
                    new_peptide = source_peptide[:mask_position] + sub_amino + source_peptide[mask_position+1:]
                    mutant_pool.append(new_peptide)
                    
        mutant_pool = sorted(list(set(mutant_pool)))
        print("Iteration-{}, mutant_pool size: {}".format(i+1, len(mutant_pool)))

        # 2. Use our model to calculate binding porbability
        #    between all peptides in mutant_pool and the given HLA
        _, _, hla_pep_tokens = seq2token(tokenizer, HLA_seq, mutant_pool, hla_max_len, pep_max_len)
        # print(hla_pep_tokens.shape)
        prob_all, score_all = [], []
        with torch.no_grad():
            start_index = 0
            end_index = batch_size if len(mutant_pool) > batch_size else len(mutant_pool)

            while end_index <= len(mutant_pool) and start_index < end_index:
                hla_pep_inputs = torch.LongTensor(hla_pep_tokens[start_index:end_index]).to(device)
                # print(hla_pep_inputs.shape)
                for ind, model in enumerate(models):
                    model_output = model(hla_pep_inputs)
                    model_score = model_output[:, 1] - model_output[:, 0]
                    model_score = model_score.cpu().detach().numpy()   # 1-D
                    if ind == 0:
                        y_score_all_ensemble = model_score
                    else:
                        y_score_all_ensemble = y_score_all_ensemble + model_score
                y_score_all_ensemble = y_score_all_ensemble / len(models)
                prob = 1 / (1+np.exp(-y_score_all_ensemble))        # sigmod, size=2
                
                prob_all.append(prob)
                score_all.append(y_score_all_ensemble)
            
                start_index = end_index
                if end_index + batch_size < len(mutant_pool):
                    end_index += batch_size
                else:
                    end_index = len(mutant_pool)

            prob_all = np.concatenate(prob_all)
            score_all = np.concatenate(score_all)
        
        # 3. Rank(score, not prob) and print "topk" messages
        if beam_width >= len(score_all):
            topk_id = np.argsort(score_all)
        else:
            topk_id = np.argsort(score_all)[-beam_width:]
        mutant_peptides = []
        mutate_table = []               # (source peptide, mutate peptide, mutate position, source amino, substitution, probablity)
        for id in topk_id:
            mutant_peptide = mutant_pool[id]
            mutant_peptides.append(mutant_peptide)

            # find a father of the mutated peptide
            for source_peptide in source_peptides:
                num_mutation, mutate_position = 0, 0
                source_amino, mutate_amino = "", ""
                assert len(source_peptide)==len(mutant_peptide)
                for position, amino in enumerate(source_peptide):
                    if amino != mutant_peptide[position]:
                        num_mutation += 1
                        mutate_position = position+1
                        source_amino = amino
                        mutate_amino = mutant_peptide[position]
                if num_mutation==1:     # means that we find its father, so no need to continue
                    break
            # record "topk" messages
            mutate_table.append((source_peptide, mutant_peptide, mutate_position, source_amino, mutate_amino, prob_all[id]))
        
        for order, mutate_info in enumerate(mutate_table):
            print("source peptide: {}, mutated peptide: {} | {} {}->{} | binding probability: {:.4f}".format(
                mutate_info[0], mutate_info[1], mutate_info[2], mutate_info[3], mutate_info[4], mutate_info[5]))
            if writein_file:
                record_file.write(f"\n>Mutate{i+1}_{order+1}\n{mutate_info[1]}")

        source_peptides = mutant_peptides       # for next iteration
        output_peptides = output_peptides+source_peptides

        if record_time:
            end_times.append(time.time())

    if writein_file:
        record_file.close()
    
    if record_time:
        end_times = np.asarray(end_times)
        return mask_pos_list, output_peptides, end_times
    else:
        return mask_pos_list, output_peptides


def get_mutated_peptides(given_HLA, init_peptide, tokenizer, models, device,
                         num_mutation=4, num_peptides=5, prob_limit=0.5,
                         writein_file=True, record_time=False, algorithm="2a", filename="supplementary_file"):
    # prepare hla_seq
    # print("HLA_seq_dict preparing")
    data_path = "/data/lujd/neoag_data/"
    hla_seq_dict = pd.read_csv(
        data_path+"main_task/HLA_sequence_dict_ABCEG.csv",
        index_col=0
        ).set_index(["HLA_name"])["clip"].to_dict()
    HLA_seq = hla_seq_dict[given_HLA]
    
    # calculate binding porbability of given peptide-HLA pair
    if record_time:
        start_time = time.time()
    _, _, hla_pep_tokens = seq2token(tokenizer, HLA_seq, init_peptide, hla_max_len, pep_max_len)
    hla_pep_inputs = torch.LongTensor(hla_pep_tokens).to(device)
    for ind, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            init_output = model(hla_pep_inputs)
            init_score = init_output[:, 1] - init_output[:, 0]
            init_score =init_score.cpu().detach().numpy()   # 1-D
            if ind == 0:
                y_score_all_ensemble = init_score
            else:
                y_score_all_ensemble = y_score_all_ensemble + init_score
    y_score_all_ensemble = y_score_all_ensemble / len(models)
    init_prob = 1 / (1+np.exp(-y_score_all_ensemble))       # sigmod
    print("given HLA: {}, given peptide: {} | binding porbability: {:.4f}".format(given_HLA, init_peptide, init_prob.item()))
    
    # design mutated peptides from given peptide
    if init_prob.item() > prob_limit:
        print("No need to mutate")
        if record_time:
            return [], [], []
        else:
            return [], []
    else:
        if algorithm == "2a":
            if record_time:
                mut_pos_list, mutant_peptides, end_times = algorithm2a(
                                                                    given_HLA, HLA_seq, init_peptide,
                                                                    tokenizer, models, device,
                                                                    num_mutation, num_peptides,
                                                                    writein_file=writein_file, record_time=record_time,
                                                                    filename=filename
                                                                    )
                run_times = end_times - start_time      # run time of each round
                return mut_pos_list, mutant_peptides, run_times
            else:
                mut_pos_list, mutant_peptides = algorithm2a(
                                                                    given_HLA, HLA_seq, init_peptide,
                                                                    tokenizer, models, device,
                                                                    num_mutation, num_peptides,
                                                                    writein_file=writein_file, record_time=record_time,
                                                                    filename=filename
                                                                    )
                return mut_pos_list, mutant_peptides
            