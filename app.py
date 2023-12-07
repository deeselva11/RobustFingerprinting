import numpy as np
import pandas as pd
import os
import hashlib
from flask import Flask, request, render_template, flash
import ot
import itertools
from flask import *
from flask_bootstrap import Bootstrap
import random
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans


app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = 'b0b4fbefdc48be27a6123605f02b6b86'

global df, X_train, X_test, y_train, y_test

apppath='C:\\Users\\Dell\\Downloads\\Project (3)\\New Folder\\Project\\Project\\CODE\\'



@app.route('/')
def home():
    return render_template('index.html')








def load_dataset(file_path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = pd.read_csv(file_path, names=column_names, sep=',\s*', engine='python')
    return data

def preprocess_data(file_path):
    # Assuming data is loaded using load_dataset function
    PER_PAGE = 500  # Set PER_PAGE to 50 records per page
    data = load_dataset(file_path)

    # Store the original DataFrame
    data_original = data.copy()

    # Encode discrete (categorical) data entries
    atts = data.columns
    for c in range(1, len(atts)):
        if data[atts[c]].dtype == 'object':
            att_list = data[atts[c]].unique()
            code_length = len(att_list)
            for i in range(1, code_length + 1):
                idx = data.index[data[atts[c]] == att_list[i - 1]].tolist()
                data.loc[idx, atts[c]] = i - 1

    # Convert all columns to numeric
    data = data.apply(pd.to_numeric, errors='ignore')

    # Quantize age attributes
    age_bins = [0, 20, 30, 40, 50, 60, 70, np.inf]
    data['age'] = pd.cut(data['age'], bins=age_bins, labels=False)

    # Remove fnlwgt attribute
    data = data.drop(['fnlwgt'], axis=1)

    # Quantize education_num attributes
    education_num_bins = [0, 4, 8, 12, np.inf]
    data['education_num'] = pd.cut(data['education_num'], bins=education_num_bins, labels=False)

    # Quantize hours_per_week attributes
    hours_per_week_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]
    data['hours_per_week'] = pd.cut(data['hours_per_week'], bins=hours_per_week_bins, labels=False)

    # Binarize capital_gain and capital_loss attributes
    data['capital_gain'] = np.where(data['capital_gain'] > 0, 1, 0)
    data['capital_loss'] = np.where(data['capital_loss'] > 0, 1, 0)



    # Save the processed DataFrame to a CSV file in the same folder
    data.to_csv('processed_data.csv', index=False)

@app.route('/view', methods=['GET', 'POST'])
def view():
    #C:\Users\Dell\Downloads\Project (3)\New Folder\Project\Project\CODE
    file_path =  apppath+ 'adult.data'

    #preprocess_data(file_path)
    PER_PAGE = 500  # Set PER_PAGE to 50 records per page
    data = load_dataset(file_path)
    data['ID'] = range(1, len(data) + 1)
    

    page = int(request.args.get('page', 1))  # Get the page number from the request

    total_records = len(data)
    num_pages = (total_records + PER_PAGE - 1) // PER_PAGE  # Calculate the total number of pages

    # Calculate the start and end indices for the current page
    start_idx = (page - 1) * PER_PAGE
    end_idx = min(start_idx + PER_PAGE, total_records)  # Ensure we don't exceed the total records

    # Get the data to display for the current page
    data_to_display = data[start_idx:end_idx]
       

   

    return render_template('view.html', data=data_to_display.to_html(classes='table table-bordered table-striped table-hover'),
                           page=page, num_pages=num_pages, selected_data=data_to_display.to_dict('records'))


@app.route('/Result', methods=['GET'])
def Result():
   
    # Assuming the Excel file is saved in the same directory as your app
    excel_file_path = apppath+'Results\\Fingerprinted_results_summary.xlsx'


    result_df = pd.read_excel(excel_file_path)


    return render_template('Result.html', result_df=result_df)

@app.route('/RobustFingerprinting', methods=['GET', 'POST'])
def RobustFingerprinting():
    return render_template('RobustFingerprinting.html')


@app.route('/DetectFingerprint', methods=['GET', 'POST'])
def DetectFingerprint():
    return render_template('DetectFingerprint.html')


@app.route('/DetectData', methods=['GET', 'POST'])
def DetectData():
    if request.method == 'POST':
        gamma = int(request.form['gamma'])
        epsilon = 1
        secret_key = 'CensusIncomeDataSet'
        sp_id = '100'

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If the file is provided and has an allowed extension (e.g., .xlsx)
        if file and file.filename.endswith('.xlsx'):
            # Read Excel file from the uploaded file
            excel_data = pd.read_excel(file, sheet_name=None)

            result_data = []

            for sheet_name, data in excel_data.items():
                f_detect = detect_fingerprint1(data, gamma, epsilon, len(data), secret_key)

                fp = sp_id_fingerprint_generate(secret_key, sp_id)
                fp = np.array(list(fp), dtype=int)
                num_match = bits_match(f_detect, fp)

                tal = 0
                for cnt in range(1, 101):
                    other_sp_ids = np.random.randint(300, 200000000, size=100)
                    match = []
                    for sp_id in other_sp_ids:
                        fp = sp_id_fingerprint_generate(secret_key, sp_id)
                        match.append(bits_match(f_detect, fp))
                    tal += np.sum(np.array(match) >= num_match)

                corrflip_innocent_accuse = tal / cnt

                # Store the results for each sheet
                result_data.append({
                    'SheetName': sheet_name,
                    'NumCompromisedBits': 128 - num_match,
                    'PercentAccusedInnocentSPs': corrflip_innocent_accuse
                })

            # Convert the results to a DataFrame
            result_df = pd.DataFrame(result_data)

            # Save the results to an Excel file
            result_df.to_excel('Fingerprinted_results_summary.xlsx', index=False)
            result_file_path = apppath+'Fingerprinted_results_summary.xlsx'
            

            # Send the file for download
            return send_file(result_file_path, as_attachment=True)
            
    return render_template('DetectFingerprint.html')


@app.route('/DetectRobustFingerprint', methods=['GET', 'POST'])
def DetectRobustFingerprint():
    if request.method == 'POST':
        gamma = int(request.form['gamma'])
        epsilon = 1
        secret_key = 'CensusIncomeDataSet'
        sp_id = '100'

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If the file is provided and has an allowed extension (e.g., .xlsx)
        if file and file.filename.endswith('.xlsx'):
            # Read Excel file from the uploaded file
            excel_data = pd.read_excel(file, sheet_name=None)

            result_data = []

            for sheet_name, data in excel_data.items():
                f_detect = detect_fingerprint1(data, gamma, epsilon, len(data), secret_key)

                fp = sp_id_fingerprint_generate(secret_key, sp_id)
                fp = np.array(list(fp), dtype=int)
                num_match = bits_match(f_detect, fp)

                tal = 0
                for cnt in range(1, 101):
                    other_sp_ids = np.random.randint(300, 200000000, size=100)
                    match = []
                    for sp_id in other_sp_ids:
                        fp = sp_id_fingerprint_generate(secret_key, sp_id)
                        match.append(bits_match(f_detect, fp))
                    tal += np.sum(np.array(match) >= num_match)

                corrflip_innocent_accuse = tal / cnt

                # Store the results for each sheet
                result_data.append({
                    'SheetName': sheet_name,
                    'NumCompromisedBits': 128 - num_match,
                    'PercentAccusedInnocentSPs': corrflip_innocent_accuse
                })

            # Convert the results to a DataFrame
            result_df = pd.DataFrame(result_data)

           

            # Save the results to an Excel file
            result_df.to_excel('VanillaFingerprinted_results_summary.xlsx', index=False)
            result_file_path = apppath+'Fingerprinted_results_summary.xlsx'
            

            # Send the file for download
            return send_file(result_file_path, as_attachment=True)
    return render_template('VanillaFingerprinting.html')

# Add a new route for correlation attacks
@app.route('/correlation_attacks', methods=['GET', 'POST'])
def correlation_attacks():
   
    return render_template('correlation_attacks.html', density=None, robustness=None)


@app.route('/GenerateDBFingerprint', methods=['GET', 'POST'])
def GenerateDBFingerprint():
    file_path =  apppath+ 'preprocessed_incomeDB.csv'
    # if request.method == 'POST':
    #     #preprocess_data(file_path)
    #     file_path =  apppath+ 'preprocessed_incomeDB.csv'
       

    data = load_csvdataset(file_path)
        
    PER_PAGE = 500
    page = int(request.args.get('page', 1))  # Get the page number from the request

    total_records = len(data)
    num_pages = (total_records + PER_PAGE - 1) // PER_PAGE  # Calculate the total number of pages

    # Calculate the start and end indices for the current page
    start_idx = (page - 1) * PER_PAGE
    end_idx = min(start_idx + PER_PAGE, total_records)  # Ensure we don't exceed the total records

    # Get the data to display for the current page
    data_to_display = data[start_idx:end_idx]
       

   

    return render_template('fingerprint.html', data=data_to_display.to_html(classes='table table-bordered table-striped table-hover'),
                           page=page, num_pages=num_pages, selected_data=data_to_display.to_dict('records'))
                
       



# Generate a fingerprint for a row
def generate_fingerprint(row):
    data_str = "".join(map(str, row))
    fingerprint = hashlib.sha256(data_str.encode()).hexdigest()
    return int(fingerprint, 16)

def calculate_sha256_hash(input_str):
    sha256 = hashlib.sha256()
    sha256.update(input_str.encode('utf-8'))
    return sha256.hexdigest()



# Function to compare joint distributions and find highly suspicious positions




def flip_lsb_of_suspicious_positions(eR, high_suspect):
    attacked_data = np.copy(eR)  # Create a copy to avoid modifying the original matrix

    for row, col in high_suspect:
        attacked_data[row, col] ^= 1  # Flip the LSB by using a bitwise XOR operation

    return attacked_data

def calculate_fingerprint_robustness(original_data, attacked_data):
    # Initialize a counter for unchanged fingerprinted records
    unchanged_records = 0

   
    for index, original_row in original_data.iterrows():
        
        if original_row['Fingerprint'] == attacked_data.at[index, 'Fingerprint']:
            unchanged_records += 1

    
    total_records = len(original_data)

    # Calculate the robustness score as a percentage
    robustness_score = (unchanged_records / total_records) * 100

    return robustness_score



def load_csvdataset(file_path):
    try:
        
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {str(e)}")
        return None

def calculate_prior_knowledge(T, s_atts_ins):
    
    atts = T.columns
    row_num, col_num = T.shape

    # Get the marginal distributions
    marginals = {}
    for i in range(1, col_num - 1):
        attribute = atts[i]
        instances = s_atts_ins[attribute]
        occurrences = [len(T[T[attribute] == ins]) for ins in instances]
        marginals[attribute] = [occ / row_num for occ in occurrences]

    # Get the joint distributions
    joints = {}
    for i in range(1, col_num - 2):
        att1 = atts[i]
        instances1 = s_atts_ins[att1]
        for j in range(i + 1, col_num - 1):
            att2 = atts[j]
            instances2 = s_atts_ins[att2]
            joint_distribution = np.zeros((len(instances1), len(instances2)))
            for idx1, ins1 in enumerate(instances1):
                row_indices1 = T[T[att1] == ins1].index
                for idx2, ins2 in enumerate(instances2):
                    joint_distribution[idx1, idx2] = len(
                        T[(T[att1].isin(row_indices1)) & (T[att2] == ins2)]
                    )
            joint_distribution /= row_num
            joints[f"{att1}with{att2}"] = joint_distribution.tolist()

    return marginals, joints




def sp_id_fingerprint_generate1(secretKey, sp_id):
    """
    Get the fingerprint of service provider (sp_id)
    secretKey: service provider's secret key
    """
    rand_str = secretKey + str(sp_id)
    fp_hex = hashlib.md5(rand_str.encode()).hexdigest()
    fp_bin = hex2bin(fp_hex)
    return fp_bin

def hex2bin(hex_str):
    bin_str = ''
    for char in hex_str:
        bin_str += bin(int(char, 16))[2:].zfill(4)
    return bin_str



def insert_fingerprint(T, gamma, epsilon, secretKey, sp_id):
    row_num, col_num = T.shape
    sp_public_number = 123

    fp = sp_id_fingerprint_generate1(secretKey, sp_id)
    L = len(fp)
    Start = 0  # In Python, indexing starts from 0
    Stop = row_num

    marked_row = []
    marked_chg_row = []
    marked_col = []
    marked_chg_col = []

    for t in range(row_num):
        primary_key_att_value = T.iloc[t, 0]
        seed = str(secretKey) + str(primary_key_att_value)
        random.seed(hash(seed))
        rnd_seq = random.sample(range(1, Stop), 5)

        if rnd_seq[0] % gamma == 0:  # if Modulo is 0, then fingerprint this tuple
            marked_row.append(t)
            att_index = rnd_seq[1] % (col_num - 2) + 1
            marked_col.append(att_index)

            att_value = T.iloc[t, att_index]
            mask_bit = rnd_seq[2] % 2
            fp_index = rnd_seq[3] % L
            fp_bit = int(fp[fp_index])
            mark_bit = mask_bit ^ fp_bit  # update mark_bit using XOR operator

            att_value_bin = bin(att_value)[2:]
            if mark_bit:
                att_value_bin = att_value_bin[:-1] + '1'
            else:
                att_value_bin = att_value_bin[:-1] + '0'

            att_value_update = int(att_value_bin, 2)
            T.iloc[t, att_index] = att_value_update

            if att_value != att_value_update:
                marked_chg_row.append(t)
                marked_chg_col.append(att_index)

    R_marked = T
    return R_marked, marked_row, marked_col, marked_chg_row, marked_chg_col

def detect_fingerprint1(T, gamma, epsilon, rnd_range, secret_key):
    L = 128
    f_vote0 = np.zeros(L)
    f_vote1 = np.zeros(L)
    
    row_num, col_num = T.shape

    for t in range(row_num):
        primary_key_att_value = T.iloc[t, 0]  # Use .iloc for integer-based indexing
        seed = [ord(char) for char in secret_key] + [primary_key_att_value]
        np.random.seed(sum(seed))
        rnd_seq = np.random.choice(range(1, rnd_range + 1), size=5, replace=False)
        
        if rnd_seq[0] % gamma == 0:
            att_index = rnd_seq[1] % (col_num - 2) + 1
            att_value = T.iloc[t, att_index]  # Use .iloc for integer-based indexing
            att_value_bin = bin(att_value)[2:]
            mark_bit = int(att_value_bin[-1])
            
            mask_bit = rnd_seq[2] % 2
            fp_bit = mark_bit ^ mask_bit
            
            fp_index = rnd_seq[3] % L
            
            if fp_bit == 0:
                f_vote0[fp_index] += 1
            else:
                f_vote1[fp_index] += 1

    f_detect = (f_vote1 / (f_vote0 + f_vote1) > 0.5).astype(float)  # Change data type to float
    f_detect[f_vote1 == f_vote0] = np.nan

    return f_detect

def fp1_insert(database, secret_key, sp_public_number):
    rand_str = str(secret_key) + str(sp_public_number)
    md5_hash = hashlib.md5(rand_str.encode()).hexdigest()
    # Pseudorandomly select an entry and alter its LSB
    random_entry = random.choice(database)
    
    # Convert random_entry to integer if it's a string
    random_entry = int(random_entry)  
    
    altered_entry = random_entry ^ int(md5_hash, 16)  # XOR operation
    
    return altered_entry

def sp_id_fingerprint_generate_v(secretKey, sp_id):
    rand_str = str(secretKey) + str(sp_id)
    md5_hash = hashlib.md5(rand_str.encode()).hexdigest()
    fp = ''.join(f'{int(char, 16):04b}' for char in md5_hash)
    return fp

def perform_columnwise_correlation_attack(eR, rounds, prior_knowledge_marginals, prior_knowledge_joints, threshold,s_atts_ins,high_suspect):
    diff_thr_list = np.array([0.01, 0.1, 0.1, 0.1, 0.06, 0.01, 0.01, 0.001, 0.001, 0.001])

  

    new_marginals, new_joints = calculate_new_joint_distributions(eR)
    select_row, select_col = obtain_suspicious_row_col(prior_knowledge_joints, new_joints, eR, diff_thr_list,s_atts_ins)

    
    high_suspect_new = find_highly_suspicious_positions(select_row, select_col)

    if not high_suspect:
        high_suspect = high_suspect_new
    else:
        high_suspect_new = [x for x in high_suspect_new if x not in high_suspect]
        high_suspect.extend(high_suspect_new)

    
    attackeddata = flip_lsb_of_suspicious_positions(eR, high_suspect)

    return attackeddata, high_suspect 

def find_highly_suspicious_positions(select_row, select_col):
    unique_row = np.unique(select_row)
    high_suspect_new = []

    for i in unique_row:
        idx = np.where(select_row == i)[0]
        suspect_attributes = np.array([select_col[j] for j in idx])

        
        attribute_counts = np.bincount(suspect_attributes.flatten())

     
        mode_attribute = np.argmax(attribute_counts)

        high_suspect_new.append([i, mode_attribute])

    return high_suspect_new

def flip_entries(eR, suspicious_positions):
    for row_index, attr_index in suspicious_positions:
        eR.iloc[row_index, attr_index] ^= 1


def obtain_suspicious_row_col(joints_public, joints_marked, R_marked, diff_thr, s_atts_ins):
    atts = joints_public.keys()
    select_row = []
    select_col = []

    #att_list = s_atts_ins.keys()
    att_list = list(s_atts_ins.keys())
    for att in atts:
        joint_pub = joints_public[att]
        joint_mar = joints_marked[att]
        joint_mar = np.array(joint_mar)
        joint_pub = np.array(joint_pub)
        joint_diff = np.abs((joint_mar - joint_pub) / joint_pub)

    for att in atts:
        joint_pub = joints_public[att]
        joint_mar = joints_marked[att]
        joint_mar = np.array(joint_mar)
        joint_pub = np.array(joint_pub)
        joint_diff = np.abs((joint_mar - joint_pub) / joint_pub)

        idx_x, idx_y = np.where(joint_diff >= diff_thr)

        att_names = att.split('with')
        att1 = att_names[0]
        att2 = att_names[1]
        idx1 = att_list.index(att1)
        idx2 = att_list.index(att2)

        for j in range(len(idx_x)):
            rows = np.where((R_marked[att1] == idx_x[j] - 1) & (R_marked[att2] == idx_y[j] - 1))
            select_row.extend(rows)
            l = len(rows[0])
            select_col.extend([[idx1 + 1, idx2 + 1]] * l)

    return select_row, select_col


# def obtain_suspicious_row_col(joints_public, joints_marked, R_marked, diff_thr_list, s_atts_ins):
    atts = list(joints_public.keys())
    select_row = []
    select_col = []
    att_list = list(s_atts_ins.keys())

    for i in range(len(atts)):
        joint_pub = joints_public[atts[i]]
        joint_mar = joints_marked[atts[i]]
        joint_diff = np.abs((joint_mar - joint_pub) / joint_pub)

        for diff_thr in diff_thr_list:
            
            idx_x, idx_y = np.where(joint_diff >= diff_thr)

            att_names = atts[i].split('with')
            att1 = att_names[0]
            att2 = att_names[1]

            idx1 = att_list.index(att1)
            idx2 = att_list.index(att2)

            for j in range(len(idx_x)):
                row_indices = np.where(
                    (R_marked[att1] == idx_x[j] - 1) & (R_marked[att2] == idx_y[j] - 1)
                )[0]

                select_row.extend(row_indices)
                l = len(row_indices)
                select_col.extend([[idx1 + 1, idx2 + 1]] * l)

    return select_row, select_col
def calculate_new_joint_distributions(eR):
    
    new_joints = {}

   
    attribute_columns = eR.columns
    for i in range(len(attribute_columns) - 1):
        for j in range(i + 1, len(attribute_columns) - 1):
            att1 = attribute_columns[i]
            att2 = attribute_columns[j]
            att_pair = f"{att1}with{att2}"

      
            joint_distribution = calculate_joint_distribution(eR, att1, att2)


            new_joints[att_pair] = joint_distribution

  
    new_marginals = calculate_marginal_distributions(eR)

    return new_marginals, new_joints

def calculate_joint_distribution(eR, att1, att2):
 
    joint_distribution = pd.crosstab(eR[att1], eR[att2], normalize='all')
    return joint_distribution.values

def calculate_marginal_distributions(eR):

    marginals = {}
    for attribute in eR.columns:
        if attribute != 'Fingerprint':
            marginal_distribution = eR[attribute].value_counts(normalize=True).sort_index()
            marginals[attribute] = marginal_distribution.values

    return marginals

def calculate_distortion_percentage(original_data, attacked_data):

    num_diff_bits = np.sum(original_data != attacked_data, axis=0)
    

    total_bits = original_data.shape[0]
    

    distortion_percentage = (num_diff_bits / total_bits) * 100
    

    average_distortion_percentage = np.mean(distortion_percentage)
    
    return average_distortion_percentage
def calculate_fingerprint_robustness(original_data, attacked_data):
    
    num_compromised_bits = np.sum(original_data != attacked_data)


    total_bits = original_data.size


    robustness = 100 - (num_compromised_bits / total_bits) * 100
    average_robustness_percentage = np.mean(robustness)
    return average_robustness_percentage


def sp_id_fingerprint_generate(secretKey, sp_id):
    """
    Get the fingerprint of service provider (sp_id)
    secretKey: service provider's secret key
    """
    rand_str = secretKey + str(sp_id)
    fp_hex = hashlib.md5(rand_str.encode()).hexdigest()
    fp_bin = hex2bin(fp_hex)
    return fp_bin

def hex2bin(hex_str):
    bin_str = ''
    for char in hex_str:
        bin_str += bin(int(char, 16))[2:].zfill(4)
    return bin_str

def insert_fingerprint(T, gamma, epsilon, secretKey, sp_id):
    row_num, col_num = T.shape
    sp_public_number = 123

    fp = sp_id_fingerprint_generate(secretKey, sp_id)
    L = len(fp)
    Start = 0  # In Python, indexing starts from 0
    Stop = row_num

    marked_row = []
    marked_chg_row = []
    marked_col = []
    marked_chg_col = []

    for t in range(row_num):
        primary_key_att_value = T.iloc[t, 0]
        seed = str(secretKey) + str(primary_key_att_value)
        random.seed(hash(seed))
        rnd_seq = random.sample(range(1, Stop), 5)

        if rnd_seq[0] % gamma == 0:  # if Modulo is 0, then fingerprint this tuple
            marked_row.append(t)
            att_index = rnd_seq[1] % (col_num - 2) + 1
            marked_col.append(att_index)

            att_value = T.iloc[t, att_index]
            mask_bit = rnd_seq[2] % 2
            fp_index = rnd_seq[3] % L
            fp_bit = int(fp[fp_index])
            mark_bit = mask_bit ^ fp_bit  # update mark_bit using XOR operator

            att_value_bin = bin(att_value)[2:]
            if mark_bit:
                att_value_bin = att_value_bin[:-1] + '1'
            else:
                att_value_bin = att_value_bin[:-1] + '0'

            att_value_update = int(att_value_bin, 2)
            T.iloc[t, att_index] = att_value_update

            if att_value != att_value_update:
                marked_chg_row.append(t)
                marked_chg_col.append(att_index)

    R_marked = T
    return R_marked, marked_row, marked_col, marked_chg_row, marked_chg_col


def insert_vanillafingerprint(T, gamma, epsilon, secretKey, sp_id):
    row_num, col_num = T.shape
    sp_public_number = 123


    fp = sp_id_fingerprint_generate_v(secretKey, sp_id)
    L = len(fp)
    Start = 0  # In Python, indexing starts from 0
    Stop = row_num

    marked_row = []
    marked_chg_row = []
    marked_col = []
    marked_chg_col = []

    for t in range(row_num):
        primary_key_att_value = T.iloc[t, 0]
        seed = str(secretKey) + str(primary_key_att_value)
        random.seed(hash(seed))
        rnd_seq = random.sample(range(1, Stop), 5)

        if rnd_seq[0] % gamma == 0:  # if Modulo is 0, then fingerprint this tuple
            marked_row.append(t)
            att_index = rnd_seq[1] % (col_num - 2) + 1
            marked_col.append(att_index)

            att_value = T.iloc[t, att_index]
            mask_bit = rnd_seq[2] % 2
            fp_index = rnd_seq[3] % L
            fp_bit = int(fp[fp_index])
            mark_bit = mask_bit ^ fp_bit  # update mark_bit using XOR operator

            att_value_bin = bin(att_value)[2:]
            if mark_bit:
                att_value_bin = att_value_bin[:-1] + '1'
            else:
                att_value_bin = att_value_bin[:-1] + '0'

            att_value_update = int(att_value_bin, 2)
            T.iloc[t, att_index] = att_value_update

            if att_value != att_value_update:
                marked_chg_row.append(t)
                marked_chg_col.append(att_index)

    R_marked = T
    return R_marked, marked_row, marked_col, marked_chg_row, marked_chg_col


def detect_fingerprint1(T, gamma, epsilon, rnd_range, secret_key):
    L = 128
    f_vote0 = np.zeros(L)
    f_vote1 = np.zeros(L)
    
    row_num, col_num = T.shape

    for t in range(row_num):
        primary_key_att_value = T.iloc[t, 0]  # Use .iloc for integer-based indexing
        seed = [ord(char) for char in secret_key] + [primary_key_att_value]
        np.random.seed(sum(seed))
        rnd_seq = np.random.choice(range(1, rnd_range + 1), size=5, replace=False)
        
        if rnd_seq[0] % gamma == 0:
            att_index = rnd_seq[1] % (col_num - 2) + 1
            att_value = T.iloc[t, att_index]  # Use .iloc for integer-based indexing
            att_value_bin = bin(att_value)[2:]
            mark_bit = int(att_value_bin[-1])
            
            mask_bit = rnd_seq[2] % 2
            fp_bit = mark_bit ^ mask_bit
            
            fp_index = rnd_seq[3] % L
            
            if fp_bit == 0:
                f_vote0[fp_index] += 1
            else:
                f_vote1[fp_index] += 1

    f_detect = (f_vote1 / (f_vote0 + f_vote1) > 0.5).astype(float)  # Change data type to float
    f_detect[f_vote1 == f_vote0] = np.nan

    return f_detect

  
def fingerprint_detection(R_marked, gamma, secretKey, sp_id):
    detected_rows = []
    detected_cols = []

    for t in range(len(R_marked)):
        primary_key_att_value = R_marked.iloc[t, 0]
        seed = str(secretKey) + str(primary_key_att_value)
        random.seed(hash(seed))
        rnd_seq = random.sample(range(1, len(R_marked)), 5)

        if rnd_seq[0] % gamma == 0:
            att_index = rnd_seq[1] % (len(R_marked.columns) - 2) + 1
            att_value = R_marked.iloc[t, att_index]
            mask_bit = rnd_seq[2] % 2
            fp_index = rnd_seq[3] % len(sp_id_fingerprint_generate(secretKey, sp_id))
            fp_bit = int(sp_id_fingerprint_generate(secretKey, sp_id)[fp_index])
            mark_bit = mask_bit ^ fp_bit

            # Check if the least significant bit of the attribute has been tampered with
            if (att_value & 1) != mark_bit:
                detected_rows.append(t)
                detected_cols.append(att_index)

    return detected_rows, detected_cols


def mass_move_all(T, plans):
    row_num, col_num = T.shape

    attack_fields_list = plans.keys()
    for attack_field in attack_fields_list:
        mass_move = plans[attack_field] * row_num

        s1 = mass_move.shape[0]

        mask = np.ones((s1, s1)) - np.eye(s1)
        mask = mask - np.tril(mask, -2) - np.triu(mask, 2)
        mass_move = mass_move * mask

        col_i = T[attack_field]

        for j in range(mass_move.shape[0]):
            idx = np.where(mass_move[j, :] >= 1)[0]
            if len(idx) == 0:
                continue
            else:
                mass_content = idx - 1  # move mass of (j-1) to mass_content
                index1 = np.where(col_i == j - 1)[0]
                perturb_idx_cumu = []
                for m in range(len(mass_content)):
                    index1 = np.setdiff1d(index1, perturb_idx_cumu)
                    perturb_idx = np.random.choice(index1, size=int(np.floor(mass_move[j, idx[m]])), replace=False)
                    perturb_idx_cumu = np.concatenate((perturb_idx_cumu, perturb_idx))
                    col_i[perturb_idx] = mass_content[m]

        T[attack_field] = col_i

    return T

def mass_move_adjacency(T, plans):
    row_num, col_num = T.shape

    attack_fields_list = plans.keys()
    for attack_field in attack_fields_list:
        mass_move = plans[attack_field] * row_num

        s1 = mass_move.shape[0]

        mask = np.ones((s1, s1)) - np.eye(s1)
        mask = mask - np.tril(mask, -2) - np.triu(mask, 2)
        mask[0, s1-1] = 1
        mask[s1-1, 0] = 1

        mass_move = mass_move * mask
        col_i = T[attack_field]

        for j in range(mass_move.shape[0]):
            idx = np.where(mass_move[j, :] >= 1)[0]
            if len(idx) == 0:
                continue
            else:
                mass_content = idx - 1  # move mass of (j-1) to mass_content
                index1 = np.where(col_i == j - 1)[0]
                perturb_idx_cumu = []
                for m in range(len(mass_content)):
                    index1 = np.setdiff1d(index1, perturb_idx_cumu)
                    perturb_idx = np.random.choice(index1, size=int(np.floor(mass_move[j, idx[m]])), replace=False)
                    perturb_idx_cumu = np.concatenate((perturb_idx_cumu, perturb_idx))
                    col_i[perturb_idx] = mass_content[m]

        T[attack_field] = col_i

    return T

def get_mass_move_plan(incomeDB,marginals_public, joints_public, marginals_marked, joints_marked, thr, lambda_val):
    num_attributes = len(marginals_public)

    plans = {}

    for p in range(num_attributes):
        for q in range(num_attributes):
            if p != q:
                J_prime_pq = joints_marked[(p, q)]
                J_public_pq = joints_public[(p, q)]

                if np.linalg.norm(J_prime_pq - J_public_pq, 'fro') > thr:
                    C_pq = calculate_joint_distribution(incomeDB, p, q)

                    plans[f'{p}_{q}'] = solve_optimal_transport(marginals_marked[p], marginals_marked[q], C_pq, lambda_val)

    return plans
def solve_optimal_transport(marginal_p, marginal_q, cost_matrix, lambda_val):
    # Normalize marginals to ensure they sum to 1
    marginal_p /= np.sum(marginal_p)
    marginal_q /= np.sum(marginal_q)

    # Solve the optimal transport problem using the Sinkhorn algorithm
    G = ot.sinkhorn(marginal_p, marginal_q, cost_matrix, reg=lambda_val)

    return G

def has_converged(G_pp_prime, max_iterations=100):
   
    return max_iterations <= 0

def Dfscol_algorithm(incomeDB, gamma, epsilon, secretKey, s_atts_ins, lambda_list, thr):
    incomeDB_sp100_gamma30, marked_row, marked_col, marked_chg_row, marked_chg_col = insert_fingerprint(incomeDB, gamma, epsilon, secretKey, 100)
       
    try:
        Q = set()
        #incomeDB_sp100_gamma30, fp_sp100, marked_row, marked_col, _, _ = insert_fingerprint(incomeDB, gamma, epsilon, secretKey, 100)
            
        M, N = incomeDB_sp100_gamma30.shape

        marginals_marked, joints_marked = calculate_prior_knowledge(incomeDB_sp100_gamma30, s_atts_ins)
        marginals_public, joints_public = calculate_prior_knowledge(incomeDB, s_atts_ins)
        num_chg_by_corrdefense = np.zeros(len(lambda_list))
        joint_diff_list = []
        Pr_Cpe = []
        Pr_Cp_prime = []

        incomeDB_sp100_gamma30_moving_history = [None] * 10

        for i, lam in enumerate(lambda_list):
            lambda_val = lam
            plans = get_mass_move_plan(incomeDB, marginals_public, joints_public, marginals_marked, joints_marked, thr,
                                       lambda_val)

            incomeDB_sp100_gamma30_mass_moved = mass_move_all(incomeDB_sp100_gamma30, plans)

            content_moved = incomeDB_sp100_gamma30_mass_moved.values
            content_marked = incomeDB_sp100_gamma30.values
            content_moved[marked_row, marked_col] = content_marked[marked_row, marked_col]
            incomeDB_sp100_gamma30_mass_moved = pd.DataFrame(content_moved, columns=incomeDB_sp100_gamma30.columns)

            incomeDB_sp100_gamma30_moving_history[i] = incomeDB_sp100_gamma30_mass_moved

            marginals_moved, joints_moved = calculate_prior_knowledge(incomeDB_sp100_gamma30_mass_moved, s_atts_ins)

            joint_diff = cum_joint_diff(joints_moved, joints_public)

            joint_diff_list.append(joint_diff)

            chg_idx = np.where(incomeDB_sp100_gamma30.values != incomeDB_sp100_gamma30_mass_moved.values)
            num_chg_by_corrdefense[i] = len(chg_idx[0])

        Q_set = set()
        for p, q in itertools.product(range(N), repeat=2):
            if p != q:
                Je_pq = calculate_joint_distribution(incomeDB_sp100_gamma30, p, q)
                J_prime_pq = joints_marked[(p, q)]

                if np.linalg.norm(J_prime_pq - Je_pq, 'fro') > thr:
                    Q_set.add(p)
                    Q_set.add(q)

        Q = list(Q_set)
        entries_to_change = np.arange(len(incomeDB_sp100_gamma30))

        for p in Q:
            Theta_pp_prime = initialize_mass_movement_cost_matrix(incomeDB_sp100_gamma30, p)
            lambda_p = initialize_tuning_parameter(Theta_pp_prime)
            G_pp_prime = np.exp(-lambda_p * Theta_pp_prime)
            k_p = len(incomeDB_sp100_gamma30.iloc[:, p].unique())
            while not has_converged(G_pp_prime):
                G_pp_prime = scale_rows_columns(G_pp_prime, Pr_Cpe, Pr_Cp_prime)

            for a in range(k_p):
                for b in range(k_p):
                    if a != b:
                        num_entries_to_change = G_pp_prime[a, b] * len(entries_to_change)
                        selected_entries = sample_entries_to_change(incomeDB_sp100_gamma30, entries_to_change, a,
                                                                    num_entries_to_change)
                        incomeDB_sp100_gamma30_mass_moved.iloc[selected_entries, p] = b

        return incomeDB_sp100_gamma30_mass_moved

    except Exception as e:
       
        return incomeDB_sp100_gamma30 

def sample_entries_to_change(incomeDB_sp100_gamma30,entries_to_change, category, num_entries_to_change):
    
    selected_entries_category = entries_to_change[incomeDB_sp100_gamma30.iloc[entries_to_change, 1] == category]

  
    selected_entries = np.random.choice(selected_entries_category, size=num_entries_to_change, replace=False)

    return selected_entries


def scale_rows_columns(G_pp_prime, Pr_Cpe, Pr_Cp_prime):
    # Scale rows based on Pr(C_pe)
    G_pp_prime_scaled_rows = G_pp_prime * Pr_Cpe.reshape((-1, 1))

    # Scale columns based on Pr(C_p_prime)
    G_pp_prime_scaled = G_pp_prime_scaled_rows * Pr_Cp_prime.reshape((1, -1))

    return G_pp_prime_scaled
def initialize_tuning_parameter(Theta_pp_prime):
    # Calculate the mean value of the non-diagonal elements in Theta_pp_prime
    non_diagonal_values = Theta_pp_prime[~np.eye(Theta_pp_prime.shape[0], dtype=bool)].mean()

    # Set the tuning parameter lambda_p
    lambda_p = 1 / non_diagonal_values

    return lambda_p
def initialize_mass_movement_cost_matrix(data, p):
    unique_values = np.unique(data[p])

    # Assuming data is a pandas DataFrame, you can get the number of unique values for each attribute
    num_unique_values = data.apply(lambda col: len(col.unique()))

    # Initialize the cost matrix with zeros
    Theta_pp_prime = np.zeros((len(unique_values), len(unique_values)))

    for i, val1 in enumerate(unique_values):
        for j, val2 in enumerate(unique_values):
            if i != j:
                # Assume you have a function calculate_cost that defines the cost between two values
                cost = calculate_cost(data, p, val1, val2)
                Theta_pp_prime[i, j] = cost

    return Theta_pp_prime

def calculate_cost(data, p, val1, val2):
    # Assuming data is a DataFrame
    entries_with_val1 = data.index[data.iloc[:, p] == val1].tolist()
    entries_with_val2 = data.index[data.iloc[:, p] == val2].tolist()

    # Calculate some cost metric based on your requirements
    cost = len(entries_with_val1) * len(entries_with_val2)

    return cost

def cum_joint_diff(joints_moved, joints_public):
    diff = 0
    for field_name in joints_public.keys():
        diff += np.linalg.norm(joints_moved[field_name] - joints_public[field_name], 'fro')

    return diff
def calculate_pairwise_relationships(data, s_atts_ins):
    communities = {}  # Dictionary to store pairwise relationships for each community

    for community_idx, attributes in enumerate(s_atts_ins):
        community_indices = [i for i in range(len(data)) if all(data.loc[i, attributes] == 1)]

        community_relationships = np.zeros((len(data), len(data)))
        for i in community_indices:
            for j in community_indices:
                if i != j:
                    # Calculate the pairwise relationship (you can customize this part based on your definition)
                    community_relationships[i, j] = calculate_pairwise_relationship(data.loc[i, attributes], data.loc[j, attributes])

        communities[community_idx] = community_relationships

    return communities

def calculate_pairwise_relationship(record1, record2):

    return np.sum(record1 * record2) / (np.linalg.norm(record1) * np.linalg.norm(record2))



def Dfsrow_algorithm(incomeDB, gamma, secretKey, s_atts_ins):
    try:
        # Obtain Se, i.e., the set of pairwise statistical relationships among individuals in each community
        incomeDB = insert_fingerprint(incomeDB, gamma,1 , secretKey, 100)
      
        Se = calculate_pairwise_relationships(incomeDB, s_atts_ins)
        C = len(Se)
        n = len(incomeDB)  # Number of individuals

        for comm_idx in range(C):
            comm = Se[comm_idx]

            # Calculate e𝑖 for each non-fingerprinted individual in the community
            e_values = []
            for i in range(n):
                if i not in comm:
                    e_i = np.sum([np.abs(Se[comm_idx][i, j] - incomeDB.iloc[i, j]) for j in comm if j != i])
                    e_values.append((i, e_i))

            # Obtain the largest ⌈𝑛𝑐𝛾⌉ 𝑒𝑖’s
            e_values.sort(key=lambda x: x[1], reverse=True)
            num_to_change = int(np.ceil(len(comm) * gamma))
            E_c = [i for i, _ in e_values[:num_to_change]]

            # Change values of selected non-fingerprinted individuals
            for i in E_c:
                most_frequent_values = {col: incomeDB.iloc[comm, col].mode()[0] for col in incomeDB.columns}
                incomeDB.iloc[i, comm] = pd.Series(most_frequent_values)

        return incomeDB
    except Exception as e:
        print(f"Error: {e}")
        return incomeDB  


@app.route('/attacks', methods=['GET', 'POST'])
def attacks():
    if request.method == 'POST':
        file_path = apppath+ 'preprocessed_incomeDB.csv'
        attack_type = request.form['attack_type']

        data = load_csvdataset(file_path)
        #size = int(request.form['size'])
        data=data.head(100)
        s_atts_ins = {}
        lambda_list = list(range(100, 1100, 100))
        thr = 0.0001
        
        for attribute in data.columns:
            
            unique_instances = data[attribute].unique()
 
            s_atts_ins[attribute] = unique_instances
        
 
        marginals, joints = calculate_prior_knowledge(data, s_atts_ins)
        
        gamma = int(request.form['gamma'])
        epsilon = 1
        secretKey = 'CensusIncomeDataSet'
        sp_id = '100'
        row_num = data.shape[0]
       
        db_file_path = apppath+ 'fp_output.csv'
        rounds_data = []
        file_path = apppath+ 'Flippeddata.xlsx'

        # Check if the file already exists
        if os.path.exists(file_path):
            # Rename the existing file to a backup name (e.g., Flippeddata_old.xlsx)
            backup_path = apppath+ 'Flippeddata_old.xlsx'
            if os.path.exists(backup_path):
              os.remove(backup_path)
            os.rename(file_path, backup_path)
       
        if attack_type == 'columnwise':
            if request.form['fingerprint_type'] == 'Fingerprinting':
           
               fingerprinted_data, marked_row, marked_col, marked_chg_row, marked_chg_col = insert_fingerprint(data, gamma, epsilon, secretKey, sp_id)
              #detected_rows, detected_cols = fingerprint_detection(fingerprinted_data, gamma, secretKey, sp_id)

            else:
               fingerprinted_data=Dfscol_algorithm(data,gamma,epsilon,secretKey,s_atts_ins,lambda_list,thr)
               #fingerprinted_data, marked_row, marked_col, marked_chg_row, marked_chg_col = insert_vanillafingerprint(data, gamma, epsilon, secretKey, sp_id)
     
            # Column-wise correlation attack
            diff_thr_list = np.array([0.01, 0.1, 0.1, 0.1, 0.06, 0.01, 0.01, 0.001, 0.001, 0.001])
            attack_rounds =  len(diff_thr_list)
            distortion_percentages = []
            distoredbits = []
            robustness_results = []
            threshold = 0.1
            R_marked_flip=fingerprinted_data
            r=6
            high_suspect = []
            excel_writer = pd.ExcelWriter(apppath+ 'Flippeddata.xlsx', engine='xlsxwriter')
            
            while r < attack_rounds:
                #excel_writer = pd.ExcelWriter('C:\\Users\\Dell\\Downloads\\Project (3)\\New Folder\\Project\\Project\\CODE\\Flippeddata.xlsx', engine='xlsxwriter')
            
                marginals_marked, joints_marked = calculate_prior_knowledge(R_marked_flip, s_atts_ins)
                diff_thr = diff_thr_list[r]
                select_row, select_col = obtain_suspicious_row_col(joints, joints_marked, R_marked_flip, diff_thr, s_atts_ins)
                select_row = np.concatenate(select_row)
                select_col = np.concatenate(select_col)
                unique_row = np.unique(select_row)
                high_suspect_new = []
                
                for i in range(len(unique_row)):
                    idx = np.where(select_row == unique_row[i])
                    sus_col = select_col[idx]
                   
                    high_suspect_new.append([unique_row[i], np.argmax(np.bincount(sus_col))])
                
                if not high_suspect:
                    high_suspect.extend(high_suspect_new)
                else:
                    high_suspect_new = list(set(tuple(row) for row in high_suspect_new) - set(tuple(row) for row in high_suspect))
                    high_suspect.extend(high_suspect_new)
                
                R_marked_flip = flipping_attack(R_marked_flip, s_atts_ins, high_suspect_new)
                
                sheet_name = f'Round_{r}'
                R_marked_flip.to_excel(excel_writer, sheet_name, index=False)
               
                
                
                r += 1
            excel_writer.close()
            
           

        elif attack_type == 'rowwise':
            if request.form['fingerprint_type'] == 'Fingerprinting':
           
               fingerprinted_data, marked_row, marked_col, marked_chg_row, marked_chg_col = insert_fingerprint(data, gamma, epsilon, secretKey, sp_id)
              #detected_rows, detected_cols = fingerprint_detection(fingerprinted_data, gamma, secretKey, sp_id)

            else:
               fingerprinted_data=Dfsrow_algorithm(data,gamma,secretKey,s_atts_ins)
               #fingerprinted_data, marked_row, marked_col, marked_chg_row, marked_chg_col = insert_vanillafingerprint(data, gamma, epsilon, secretKey, sp_id)
     
            Dfsrow_algorithm
            excel_writer = pd.ExcelWriter(apppath+ 'Flippeddata.xlsx', engine='xlsxwriter')
            
            col_num = data.shape[1]
            db = data.iloc[:, 1:col_num - 1].values
            db_mark = fingerprinted_data.iloc[:, 1:col_num - 1].values

            # Define the number of communities and seed for reproducibility
            comm = 10
            np.random.seed(123)

            # Perform K-means clustering to identify community memberships
            affiliation = KMeans(n_clusters=comm, random_state=123).fit_predict(db)

           
            gamma = 30
            sus = []

            for j in range(1, comm + 1):
                individual_id = np.where(affiliation == j)[0]
                num_individual = len(individual_id)
                individual_of_comm_i = db[individual_id, :]
                D_og = squareform(pdist(individual_of_comm_i, 'hamming')) * 13

                D_og = np.exp(-D_og / 13) - np.eye(num_individual)

                individual_of_comm_i_mark = db_mark[np.where(affiliation == j), :][0]
                D_mark = squareform(pdist(individual_of_comm_i_mark, 'hamming')) * 13
                D_mark = np.exp(-D_mark / 13) - np.eye(num_individual)

                abs_diff = np.abs(D_og - D_mark)
                id_diff = np.argsort(np.sum(abs_diff, axis=1))[::-1]

                if len(id_diff) > 0:
                    num_sus = int(np.ceil(num_individual / 10))
                    sus.extend(individual_id[id_diff[:num_sus]])

            high_suspect = np.column_stack((np.tile(sus, 13), np.repeat(range(2, col_num), len(sus))))
            R_marked_flip = flipping_attack(fingerprinted_data, s_atts_ins, high_suspect)
            R_marked_flip.to_excel(excel_writer, 'sheet_1', index=False)

            excel_writer.close()
        elif attack_type == 'Integrated':
            if request.form['fingerprint_type'] == 'Fingerprinting':
           
               fingerprinted_data, marked_row, marked_col, marked_chg_row, marked_chg_col = insert_fingerprint(data, gamma, epsilon, secretKey, sp_id)
              #detected_rows, detected_cols = fingerprint_detection(fingerprinted_data, gamma, secretKey, sp_id)

            else:
               fingerprinted_data=Dfscol_algorithm(data,gamma,epsilon,secretKey,s_atts_ins,lambda_list,thr)
             
            col_num = data.shape[1]
            db = data.iloc[:, 1:col_num - 1].values
            db_mark = fingerprinted_data.iloc[:, 1:col_num - 1].values

            # Define the number of communities and seed for reproducibility
            comm = 10
            np.random.seed(123)

            # Perform K-means clustering to identify community memberships
            affiliation = KMeans(n_clusters=comm, random_state=123).fit_predict(db)

           
            gamma = 30
            sus = []

            for j in range(1, comm + 1):
                individual_id = np.where(affiliation == j)[0]
                num_individual = len(individual_id)
                individual_of_comm_i = db[individual_id, :]
                D_og = squareform(pdist(individual_of_comm_i, 'hamming')) * 13

                D_og = np.exp(-D_og / 13) - np.eye(num_individual)

                individual_of_comm_i_mark = db_mark[np.where(affiliation == j), :][0]
                D_mark = squareform(pdist(individual_of_comm_i_mark, 'hamming')) * 13
                D_mark = np.exp(-D_mark / 13) - np.eye(num_individual)

                abs_diff = np.abs(D_og - D_mark)
                id_diff = np.argsort(np.sum(abs_diff, axis=1))[::-1]

                if len(id_diff) > 0:
                    num_sus = int(np.ceil(num_individual / 10))
                    sus.extend(individual_id[id_diff[:num_sus]])

            high_suspect = np.column_stack((np.tile(sus, 13), np.repeat(range(2, col_num), len(sus))))
            R_marked_flip = flipping_attack(fingerprinted_data, s_atts_ins, high_suspect)

            diff_thr_list = np.array([0.01, 0.1, 0.1, 0.1, 0.06, 0.01, 0.01, 0.001, 0.001, 0.001])
            attack_rounds =  len(diff_thr_list)
            distortion_percentages = []
            distoredbits = []
            robustness_results = []
            threshold = 0.1
            
            r=1
            high_suspect = []
            excel_writer = pd.ExcelWriter(apppath+ 'Flippeddata.xlsx', engine='xlsxwriter')
            
            while r < attack_rounds:
                #excel_writer = pd.ExcelWriter('C:\\Users\\Dell\\Downloads\\Project (3)\\New Folder\\Project\\Project\\CODE\\Flippeddata.xlsx', engine='xlsxwriter')
            
                marginals_marked, joints_marked = calculate_prior_knowledge(R_marked_flip, s_atts_ins)
                diff_thr = diff_thr_list[r]
                select_row, select_col = obtain_suspicious_row_col(joints, joints_marked, R_marked_flip, diff_thr, s_atts_ins)
                select_row = np.concatenate(select_row)
                select_col = np.concatenate(select_col)
                unique_row = np.unique(select_row)
                high_suspect_new = []
                
                for i in range(len(unique_row)):
                    idx = np.where(select_row == unique_row[i])
                    sus_col = select_col[idx]
                   
                    high_suspect_new.append([unique_row[i], np.argmax(np.bincount(sus_col))])
                
                if not high_suspect:
                    high_suspect.extend(high_suspect_new)
                else:
                    high_suspect_new = list(set(tuple(row) for row in high_suspect_new) - set(tuple(row) for row in high_suspect))
                    high_suspect.extend(high_suspect_new)
                
                R_marked_flip = flipping_attack(R_marked_flip, s_atts_ins, high_suspect_new)
                
              
                
                sheet_name = f'Round_{r}'
                R_marked_flip.to_excel(excel_writer, sheet_name, index=False)
               
                
       
                
                r += 1
            excel_writer.close()
           
        PER_PAGE = 500
        page = int(request.args.get('page', 1))  # Get the page number from the request

        total_records = len(data)
        num_pages = (total_records + PER_PAGE - 1) // PER_PAGE  # Calculate the total number of pages

        # Calculate the start and end indices for the current page
        start_idx = (page - 1) * PER_PAGE
        end_idx = min(start_idx + PER_PAGE, total_records)  # Ensure we don't exceed the total records

        # Get the data to display for the current page
        data_to_display = data[start_idx:end_idx]
       

   

        return render_template('fingerprint.html', data=data_to_display.to_html(classes='table table-bordered table-striped table-hover'),
                           page=page, num_pages=num_pages, selected_data=data_to_display.to_dict('records'))
                
        

def bits_match(fp_detect, fp):
    num_match = np.sum(fp_detect == fp)
    return num_match
def flipping_attack(R_marked, s_atts_ins, high_suspect):
    # Extract the list of fingerprinted attribute names
    fp_att_list = R_marked.columns[1:-1].tolist()  # Exclude the id and label columns

    flip_length = len(high_suspect)

    for i in range(flip_length):
        row = high_suspect[i][0]
        col = high_suspect[i][1] - 1  # Adjust the column index

        # Check if col is a valid index
        if col < 0 or col >= len(fp_att_list):
            print(f"Invalid column index {col}")
            continue

        all_states = s_atts_ins[fp_att_list[col]]
        sus_entry = R_marked.iloc[row, high_suspect[i][1]]
        sus_entry_bin = bin(sus_entry)[2:]  # Convert the entry to binary
        mark_bit_m = sus_entry_bin[-1]

        if mark_bit_m == '1':
            new_sus_entry_bin = sus_entry_bin
            new_sus_entry_bin = new_sus_entry_bin[:-1] + '0'
            new_sus_entry = int(new_sus_entry_bin, 2)
        else:
            new_sus_entry_bin = sus_entry_bin
            new_sus_entry_bin = new_sus_entry_bin[:-1] + '1'
            new_sus_entry = int(new_sus_entry_bin, 2)
            if new_sus_entry > max(all_states):
                new_sus_entry -= 2

        R_marked.iat[row, high_suspect[i][1]] = new_sus_entry

    R_marked_flip = R_marked.copy()
    
    return R_marked_flip






def bits_match(fp_detect, fp):
    """
    Calculate the number of matched bits given an extracted fingerprint and a true fingerprint.
    
    Args:
        fp_detect: Extracted fingerprint
        fp: True fingerprint
        
    Returns:
        num_match: Number of matched bits
    """
    num_match = np.sum(fp_detect == fp)
    return num_match
           







     
           
# def compute_relationship_measure(data1, data2):

#     return relationship_measure


# def calculate_discrepancy(new_relationships, prior_relationships):

#     return discrepancy

def manipulate_individual_data(attacked_data, individual_index):

    return attacked_data

def perform_integrated_correlation_attack(eR, rounds, prior_knowledge_marginals, s_atts_ins):
    
    attacked_data = eR  



    return attacked_data



if __name__== '__main__':
    app.run(debug=True)