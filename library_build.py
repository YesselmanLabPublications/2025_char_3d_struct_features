import re
import os
import pandas as pd
import editdistance
import random
from vienna import fold

df = pd.read_csv('motif_sequences.csv')

# Initialize variables
pool = []
pool_motifs = []
pool_m_ss = []
usable_seq = []
usable_ss = []
usable_motifs = []
usable_m_ss = []
seq_len = []
ens_def = []
edit_dis = []

max_count = 100  # maximum number that a motif can appear in constructs
desired_sequences = 7500  # Desired number of sequences

while len(usable_seq) < desired_sequences:
    selected_rows = []  # To store the selected columns
    selected_count = {}   # To keep track of the count of each selected column
    selected_motif = []
    selected_ss = []

    hairpin = list('GCGAGUAGC')
    hairpin_ss = list('((.....))')
    RNA_bases = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
    five_prime = list("GGGCUUCGGCCCA")
    five_prime_ss = list("((((....)))).")
    three_prime = list("ACAAAGAAACAACAACAACAAC")
    three_prime_ss = list("......................")

    items = list(RNA_bases.items())
    full_seq_right = []
    full_seq_left = []
    full_seq_right_ss = []
    full_seq_left_ss = []

    # Generate complementary hairpin sets with two random base pairs
    hairpin_set1 = []
    hairpin_set2 = []
    for k in range(2):
        random.shuffle(items)
        position = random.randint(0, len(items) - 1)
        key, value = items[position]
        hairpin_set1.append(key)
        hairpin_set2.append(value)
        hairpin_st2_rev = hairpin_set2[::-1]

    num_rows_to_select = random.randint(5, 7)
    while len(selected_rows) < num_rows_to_select:
        available_rows = [row for row in df.index if row not in selected_rows]
        if not available_rows:
            break
        random_row = random.choice(available_rows)

        # Check if the row can be selected without exceeding the count limit
        if random_row in selected_count and selected_count[random_row] >= max_count:
            continue 

        selected_rows.append(random_row)

        # Update the count for the selected row
        if random_row in selected_count:
            selected_count[random_row] += 1
        else:
            selected_count[random_row] = 1

    for row in selected_rows:
        seq_value = df.loc[row, 'motif_seq']
        ss_value = df.loc[row, 'motif_ss']
        selected_motif.append(seq_value)
        selected_ss.append(ss_value)

        set1 = []
        set2 = []
        set1_ss = []
        set2_ss = []

        # Generate three pairs of complementary bases for helices
        for j in range(3):
            random.shuffle(items)
            position1 = random.randint(0, len(items) - 1)
            key, value = items[position1]
            set1.append(key)
            st1 = ''.join(set1)

            set2.append(value)
            st2 = ''.join(set2)
            st2_rev = st2[::-1]

        st1_ss = '((('
        st2_ss = ')))'
        seq_value1 = seq_value.split('&')[0]
        seq_value2 = seq_value.split('&')[1]
        ss_value1 = ss_value.split('&')[0]
        ss_value2 = ss_value.split('&')[1]

        sq1 = st1 + seq_value1
        full_seq_left.append(sq1)
        ss1 = st1_ss + ss_value1
        full_seq_left_ss.append(ss1)
        sq2 = seq_value2 + st2_rev
        full_seq_right.insert(0, sq2)
        ss2 = ss_value2 + st2_ss
        full_seq_right_ss.insert(0, ss2)

    seq = five_prime + full_seq_left + hairpin_set1 + hairpin + hairpin_st2_rev + full_seq_right + three_prime
    ss = five_prime_ss + full_seq_left_ss + ['(', '('] + hairpin_ss + [')', ')'] + full_seq_right_ss + three_prime_ss

    ss_str = ''.join(ss)
    full_ss = ''.join(ss)
    full_seq = ''.join(seq)

    # Check the length condition
    if len(full_seq) <= 150:
        continue 
    
    # Check the length difference condition
    if usable_seq:
        min_length = min(len(seq) for seq in usable_seq)
        if not (len(full_seq) < min_length * 1.1):
            continue

    full_ss_RNAfold = fold(full_seq).dot_bracket

    if full_ss == full_ss_RNAfold:
        pool.append(full_seq)
        pool_motifs.append(selected_motif)
        pool_m_ss.append(selected_ss)

    for i, (p1, m1, s1) in enumerate(zip(pool, pool_motifs, pool_m_ss)):
        folded_p1 = fold(p1) 
        ens_defect_p1 = folded_p1.ens_defect

        for p2 in pool[i:]:
            y = editdistance.eval(p1, p2)
            if y > 20 and ens_defect_p1 <= 5:
                if p1 not in usable_seq:
                    print(p1)
                    usable_motifs.append(m1)
                    usable_m_ss.append(s1)
                    usable_seq.append(p1)
                    usable_ss.append(folded_p1.dot_bracket)
                    s_len = len(p1)
                    seq_len.append(s_len)
                    ens_def.append(ens_defect_p1)
                    edit_dis.append(y)
                break 

        # Stop once the desired number of sequences is reached
        if len(usable_seq) >= desired_sequences:
            break

df_final = pd.DataFrame()
df_final['seq'] = usable_seq
df_final['ss'] = usable_ss
df_final['motifs'] = usable_motifs
df_final['motifs_ss'] = usable_m_ss
df_final['len'] = seq_len
df_final['ens_defect'] = ens_def
df_final['edit_distance'] = edit_dis
df_final.to_json('pdb_library.json', orient="records")
