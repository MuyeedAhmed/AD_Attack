import pandas as pd
import os


def RemoveDuplicates(file_path):
    df = pd.read_csv(file_path)
    df_no_duplicates = df.drop_duplicates()
    df_no_duplicates.to_csv(file_path, index=False)


def mergeFiles():
    file_a_path = 'IF.csv'
    file_b_path = 'LOF.csv'
    file_c_path = 'OCSVM.csv'
    
    df_a = pd.read_csv(file_a_path)
    df_b = pd.read_csv(file_b_path)
    df_c = pd.read_csv(file_c_path)
    
    df_a['Algorithm'] = 'IF'
    df_b['Algorithm'] = 'LOF'
    df_c['Algorithm'] = 'OCSVM'
    
    merged_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
    merged_df['Algorithm'] = merged_df.groupby('Repository URL')['Algorithm'].transform(lambda x: ';'.join(x))
    
    merged_df = merged_df.drop_duplicates(subset='Repository URL')
    
    output_file_path = 'AD.csv'  
    merged_df.to_csv(output_file_path, index=False)



def MergeAllFilesOfAFolder(folder_path):
    dfs = []
    
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Path to save the merged CSV file
    merged_csv_path = folder_path+'/ALL.csv'
    
    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(merged_csv_path, index=False)



folder_path = 'List_Cpp'
MergeAllFilesOfAFolder(folder_path)