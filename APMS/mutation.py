# return to parent directory
import sys 
if ".." not in sys.path:
    sys.path.append("..")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tape import TAPETokenizer, ProteinBertConfig
from model_ft import meanTAPE

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

def algorithm1a( 
    given_HLA, HLA_seq, init_peptide, 
    tokenizer, model, device,
    iteration, beam_width,
    filename="mutant_peptides2"
    ):
    record_file = open(f"./{filename}/al1a_{given_HLA}_len{len(init_peptide)}_iter{iteration}", "w")
    record_file.write(f">Source\n{init_peptide}")

    if iteration > len(init_peptide):
        iteration = len(init_peptide)

    batch_size = 64
    source_peptides = [init_peptide]
    output_peptides = []

    for i in range(iteration):
        # 1. Make a pool of candidate mutated peptides
        mutant_pool = []
        for source_peptide in source_peptides:
            for ind, amino in enumerate(source_peptide):
                if amino == init_peptide[ind]:              # find non-mutated position against given peptide
                    for sub_amino in amino_acid_list:       # replace amino at the position
                        if sub_amino != amino:
                            new_peptide = source_peptide[:ind] + sub_amino + source_peptide[ind+1:]
                            mutant_pool.append(new_peptide)
                # else:
                #     print(ind+1)
            mutant_pool = sorted(list(set(mutant_pool)))
        print("Iteration-{}, mutant_pool size: {}".format(i+1, len(mutant_pool)))

        # 2. Use our finetuned TAPE to calculate binding porbability
        # between all peptides in mutant_pool and the given HLA
        _, _, hla_pep_tokens = seq2token(tokenizer, HLA_seq, mutant_pool, hla_max_len, pep_max_len)
        # print(hla_pep_tokens.shape)
        prob_all, score_all = [], []
        with torch.no_grad():
            start_index = 0
            end_index = batch_size if len(mutant_pool) > batch_size else len(mutant_pool)

            while end_index <= len(mutant_pool) and start_index < end_index:
                hla_pep_inputs = torch.LongTensor(hla_pep_tokens[start_index:end_index]).to(device)
                # print(hla_pep_inputs.shape)
                model_output = model(hla_pep_inputs)
                prob = nn.Softmax(dim=1)(model_output)[:, 1].cpu().detach().numpy() # 1-D
                score = (model_output[:, 1] - model_output[:, 0]).cpu().detach().numpy()
                prob_all.append(prob)
                score_all.append(score)

                start_index = end_index
                if end_index + batch_size < len(mutant_pool):
                    end_index += batch_size
                else:
                    end_index = len(mutant_pool)

            prob_all = np.concatenate(prob_all)
            score_all = np.concatenate(score_all)
        
        # 3. Rank(score, not prob) and print "topk" messages
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
            record_file.write(f"\n>Mutate{i+1}_{order+1}\n{mutate_info[1]}")

        source_peptides = mutant_peptides       # for next iteration
        output_peptides = output_peptides+source_peptides

    record_file.close()
    return output_peptides


def algorithm1b( 
    given_HLA, HLA_seq, init_peptide, 
    tokenizer, model, device,
    iteration, beam_width,
    filename="mutant_peptides"
    ):
    record_file = open(f"./{filename}/al1b_{given_HLA}_len{len(init_peptide)}_iter{iteration}", "w")
    record_file.write(f">Source\n{init_peptide}")

    if iteration > len(init_peptide):
        iteration = len(init_peptide)

    batch_size = 64
    source_peptides = [init_peptide]
    output_peptides = []

    for i in range(iteration):
        # 1. Make a pool of candidate mutated peptides
        mutant_pool = []
        flag = 0
        frozen_position = []
        for source_peptide in source_peptides:
            for ind, amino in enumerate(source_peptide):
                if amino == init_peptide[ind]:              # find non-mutated position against given peptide
                    for sub_amino in amino_acid_list:       # replace amino at the position
                        if sub_amino != amino:
                            new_peptide = source_peptide[:ind] + sub_amino + source_peptide[ind+1:]
                            mutant_pool.append(new_peptide)
                else:
                    if flag == 0:
                        frozen_position.append(ind)
            flag = 1
        print("Iteration-{}, mutant_pool size: {}".format(i+1, len(mutant_pool)))
        # print(frozen_position)

        # 2. Use our finetuned TAPE to calculate binding porbability
        # between all peptides in mutant_pool and the given HLA
        _, _, hla_pep_tokens = seq2token(tokenizer, HLA_seq, mutant_pool, hla_max_len, pep_max_len)
        # print(hla_pep_tokens.shape)
        prob_all, score_all = [], []
        with torch.no_grad():
            start_index = 0
            end_index = batch_size if len(mutant_pool) > batch_size else len(mutant_pool)

            while end_index <= len(mutant_pool) and start_index < end_index:
                hla_pep_inputs = torch.LongTensor(hla_pep_tokens[start_index:end_index]).to(device)
                # print(hla_pep_inputs.shape)
                model_output = model(hla_pep_inputs)
                prob = nn.Softmax(dim=1)(model_output)[:, 1].cpu().detach().numpy() # 1-D
                score = (model_output[:, 1] - model_output[:, 0]).cpu().detach().numpy()
                prob_all.append(prob)
                score_all.append(score)

                start_index = end_index
                if end_index + batch_size < len(mutant_pool):
                    end_index += batch_size
                else:
                    end_index = len(mutant_pool)

            prob_all = np.concatenate(prob_all)
            score_all = np.concatenate(score_all)
        
        # 3. Rank and choose topk at each position, then average and choose the best position
        sorted_id = np.argsort(-score_all)
        id_table = np.zeros((len(init_peptide), beam_width), dtype=int)
        prob_table = np.zeros((len(init_peptide), beam_width))
        num_record = np.zeros(len(init_peptide), dtype=int)
        for pos in frozen_position:
            num_record[pos] = beam_width        # when sum(num_record==beam_width)==len(init_peptide), stop searching
        
        for id in sorted_id:
            mutant_peptide = mutant_pool[id]
            
            # find a father of the mutated peptide
            num_mutation, mutate_position = 0, 0
            for source_peptide in source_peptides:
                for position, amino in enumerate(source_peptide):
                    if amino != mutant_peptide[position]:
                        num_mutation += 1
                        mutate_position = position
                if num_mutation==1:     # means that we find its father, so no need to continue
                    break
            
            # record
            if num_record[mutate_position] < beam_width:
                id_table[mutate_position, num_record[mutate_position]] = id
                prob_table[mutate_position, num_record[mutate_position]] = prob_all[id]
                num_record[mutate_position] += 1
            
            # when to stop
            if np.sum(num_record==beam_width)==len(init_peptide):
                break

        prob_table = np.mean(prob_table, axis=1)
        # print(prob_table)
        best_position = np.argsort(prob_table)[-1]
        # print(best_position, id_table[best_position])

        # 4. print "topk" messages
        mutant_peptides = []
        mutate_table = []               # (source peptide, mutate peptide, mutate position, source amino, substitution, probablity)
        for id in id_table[best_position]:
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
            record_file.write(f"\n>Mutate{i+1}_{order+1}\n{mutate_info[1]}")

        source_peptides = mutant_peptides       # for next iteration
        output_peptides = output_peptides+source_peptides

    record_file.close()
    return output_peptides


def algorithm2a( 
    given_HLA, HLA_seq, init_peptide, 
    tokenizer, model, device,
    iteration, beam_width,
    filename="mutant_peptides"
    ):
    record_file = open(f"./{filename}/al2a_{given_HLA}_len{len(init_peptide)}_iter{iteration}", "w")
    record_file.write(f">Source\n{init_peptide}")

    if iteration > len(init_peptide):
        iteration = len(init_peptide)

    batch_size = 16
    source_peptides = [init_peptide]
    output_peptides = []

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
                        model_output = model(hla_pep_inputs)
                        prob = nn.Softmax(dim=1)(model_output)[:, 1].cpu().detach().numpy() # 1-D
                        # print(prob)
                        saliency = prob[1] - prob[0]        # masked - origin
                        # print(saliency)
                        saliency_all.append(saliency)
                else:
                    # print(ind+1)
                    saliency_all.append(-1)
            
            # (2) saliency ranking
            saliency_all = np.array(saliency_all)
            # print(saliency_all, len(saliency_all))
            mask_position = np.argsort(saliency_all)[-1]    # best position to be replaced
            # print(mask_position+1)

            # (3) replace amino at the mask_position
            for sub_amino in amino_acid_list:
                if sub_amino != source_peptide[mask_position]:
                    new_peptide = source_peptide[:mask_position] + sub_amino + source_peptide[mask_position+1:]
                    mutant_pool.append(new_peptide)
                    
        mutant_pool = sorted(list(set(mutant_pool)))
        print("Iteration-{}, mutant_pool size: {}".format(i+1, len(mutant_pool)))

        # 2. Use our finetuned TAPE to calculate binding porbability
        # between all peptides in mutant_pool and the given HLA
        _, _, hla_pep_tokens = seq2token(tokenizer, HLA_seq, mutant_pool, hla_max_len, pep_max_len)
        # print(hla_pep_tokens.shape)
        prob_all, score_all = [], []
        with torch.no_grad():
            start_index = 0
            end_index = batch_size if len(mutant_pool) > batch_size else len(mutant_pool)

            while end_index <= len(mutant_pool) and start_index < end_index:
                hla_pep_inputs = torch.LongTensor(hla_pep_tokens[start_index:end_index]).to(device)
                # print(hla_pep_inputs.shape)
                model_output = model(hla_pep_inputs)
                prob = nn.Softmax(dim=1)(model_output)[:, 1].cpu().detach().numpy() # 1-D
                score = (model_output[:, 1] - model_output[:, 0]).cpu().detach().numpy()
                prob_all.append(prob)
                score_all.append(score)
            
                start_index = end_index
                if end_index + batch_size < len(mutant_pool):
                    end_index += batch_size
                else:
                    end_index = len(mutant_pool)

            prob_all = np.concatenate(prob_all)
            score_all = np.concatenate(score_all)
        
        # 3. Rank(score, not prob) and print "topk" messages
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
            record_file.write(f"\n>Mutate{i+1}_{order+1}\n{mutate_info[1]}")

        source_peptides = mutant_peptides       # for next iteration
        output_peptides = output_peptides+source_peptides

    record_file.close()
    return output_peptides


def algorithm2b(
    given_HLA, HLA_seq, init_peptide, 
    tokenizer, model, device,
    iteration, beam_width,
    filename="mutant_peptides"
    ):
    record_file = open(f"./{filename}/al2b_{given_HLA}_len{len(init_peptide)}_iter{iteration}", "w")
    record_file.write(f">Source\n{init_peptide}")

    if iteration > len(init_peptide):
        iteration = len(init_peptide)
    
    batch_size = 16
    source_peptides = [init_peptide]
    output_peptides = []

    for i in range(iteration):
        # 1. Make a pool of candidate mutated peptides
        # (1) replace amino with <unk> and calculate saliency
        mutant_pool = []
        saliency_all = []
        for source_peptide in source_peptides:
            saliency_single = []
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
                        model_output = model(hla_pep_inputs)
                        prob = nn.Softmax(dim=1)(model_output)[:, 1].cpu().detach().numpy() # 1-D
                        # print(prob)
                        saliency = prob[1] - prob[0]        # masked - origin
                        # print(saliency)
                        saliency_single.append(saliency)
                else:
                    # print(ind+1)
                    saliency_single.append(-1)
            # print(saliency_single)
            saliency_all.append(saliency_single)
            
        # (2) calculate average saliency and rank
        saliency_all = np.array(saliency_all)
        # print(saliency_all, saliency_all.shape)
        saliency_all = np.mean(saliency_all, axis=0)
        # print(saliency_all, saliency_all.shape)
        mask_position = np.argsort(saliency_all)[-1]    # best position to be replaced
        # print(mask_position+1)

        # (3) replace amino at the mask_position
        for source_peptide in source_peptides:
            for sub_amino in amino_acid_list:
                if sub_amino != source_peptide[mask_position]:
                    new_peptide = source_peptide[:mask_position] + sub_amino + source_peptide[mask_position+1:]
                    mutant_pool.append(new_peptide)
                    
        mutant_pool = sorted(list(set(mutant_pool)))
        print("Iteration-{}, mutant_pool size: {}".format(i+1, len(mutant_pool)))

        # 2. Use our finetuned TAPE to calculate binding porbability
        # between all peptides in mutant_pool and the given HLA
        _, _, hla_pep_tokens = seq2token(tokenizer, HLA_seq, mutant_pool, hla_max_len, pep_max_len)
        # print(hla_pep_tokens.shape)
        prob_all, score_all = [], []
        with torch.no_grad():
            start_index = 0
            end_index = batch_size if len(mutant_pool) > batch_size else len(mutant_pool)

            while end_index <= len(mutant_pool) and start_index < end_index:
                hla_pep_inputs = torch.LongTensor(hla_pep_tokens[start_index:end_index]).to(device)
                # print(hla_pep_inputs.shape)
                model_output = model(hla_pep_inputs)
                prob = nn.Softmax(dim=1)(model_output)[:, 1].cpu().detach().numpy() # 1-D
                score = (model_output[:, 1] - model_output[:, 0]).cpu().detach().numpy()
                prob_all.append(prob)
                score_all.append(score)
            
                start_index = end_index
                if end_index + batch_size < len(mutant_pool):
                    end_index += batch_size
                else:
                    end_index = len(mutant_pool)

            prob_all = np.concatenate(prob_all)
            score_all = np.concatenate(score_all)
        
        # 3. Rank(score, not prob) and print "topk" messages
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
            record_file.write(f"\n>Mutate{i+1}_{order+1}\n{mutate_info[1]}")

        source_peptides = mutant_peptides       # for next iteration
        output_peptides = output_peptides+source_peptides

    record_file.close()
    return output_peptides


def get_mutated_peptides(given_HLA, init_peptide, device, num_mutation=2, num_peptides=5,
                         algorithm="1a", filename="mutant_peptides"):
    # prepare hla_seq
    print("HLA_seq_dict preparing")
    data_path = "/data/lujd/neoag_data/"
    hla_seq_dict = pd.read_csv(
        data_path+"main_task/HLA_sequence_dict_ABCEG.csv",
        index_col=0
        ).set_index(["HLA_name"])["clip"].to_dict()
    HLA_seq = hla_seq_dict[given_HLA]
    
    # prepare model
    model_path = "/data/lujd/neoag_model/main_task/TAPE_ft/cat_mean_2mlp/"
    model_name = "main_finetune_plm_tape_B32_LR3e-05_seq_clip_fold4_ep51_221104.pkl"

    print("Model preparing")
    tokenizer = TAPETokenizer(vocab='iupac')
    tape_config = ProteinBertConfig.from_pretrained('bert-base')
    model = meanTAPE(tape_config, "2mlp").to(device)
    model.load_state_dict(torch.load(model_path + model_name), strict = True)
    model.eval()

    # calculate binding porbability between given HLA and peptide
    _, _, hla_pep_tokens = seq2token(tokenizer, HLA_seq, init_peptide, hla_max_len, pep_max_len)
    with torch.no_grad():
        hla_pep_inputs = torch.LongTensor(hla_pep_tokens).to(device)
        init_prob = model(hla_pep_inputs)
        init_prob = nn.Softmax(dim=1)(init_prob)[:, 1].cpu().detach().numpy()   # 1-D
    print("given HLA: {}, given peptide: {} | binding porbability: {:.4f}".format(given_HLA, init_peptide, init_prob.item()))
    
    # design mutated peptides from given peptide
    if init_prob.item()>0.5:
        print("No need to mutate")
    else:
        if algorithm == "1a":
            print("************** Run algorithm-1a **************")
            mutant_peptides = algorithm1a(
                                        given_HLA, HLA_seq, init_peptide,
                                        tokenizer, model, device,
                                        num_mutation, num_peptides,
                                        filename=filename
                                        )
            print("-------------- Algorithm-1a done --------------")
        elif algorithm == "1b":
            print("************** Run algorithm-1b **************")
            mutant_peptides = algorithm1b(
                                        given_HLA, HLA_seq, init_peptide,
                                        tokenizer, model, device,
                                        num_mutation, num_peptides,
                                        filename=filename
                                        )
            print("-------------- Algorithm-1b done --------------")
        elif algorithm == "2a":
            print("************** Run algorithm-2a **************")
            mutant_peptides = algorithm2a(
                                        given_HLA, HLA_seq, init_peptide,
                                        tokenizer, model, device,
                                        num_mutation, num_peptides,
                                        filename=filename
                                        )
            print("-------------- Algorithm-2a done --------------")
        elif algorithm == "2b":
            print("************** Run algorithm-2b **************")
            mutant_peptides = algorithm2b(
                                        given_HLA, HLA_seq, init_peptide,
                                        tokenizer, model, device,
                                        num_mutation, num_peptides,
                                        filename=filename
                                        )
            print("-------------- Algorithm-2b done --------------")
    
        return mutant_peptides