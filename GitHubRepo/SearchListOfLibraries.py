import os
import pandas as pd
import subprocess



def RunBash(GITHUB_TOKEN, lib):    
    try:
        subprocess.run(["./SearchRepo.sh", GITHUB_TOKEN, lib], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the Bash script: {e}")

    try:
        subprocess.run(["./SearchStars.sh", GITHUB_TOKEN], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the Bash script: {e}")

def RemoveDuplicates(file_path, lib):
    df = pd.read_csv(file_path)
    df_no_duplicates = df.drop_duplicates()
    df_no_duplicates["Algorithm"] = lib
    df_no_duplicates.to_csv("List/"+lib+".csv", index=False)
    os.remove(file_path)

# def mergeFiles():
#     file_a_path = 'IF.csv'
#     file_b_path = 'LOF.csv'
#     file_c_path = 'OCSVM.csv'
    
#     df_a = pd.read_csv(file_a_path)
#     df_b = pd.read_csv(file_b_path)
#     df_c = pd.read_csv(file_c_path)
    
#     df_a['Algorithm'] = 'IF'
#     df_b['Algorithm'] = 'LOF'
#     df_c['Algorithm'] = 'OCSVM'
    
#     merged_df = pd.concat([df_a, df_b, df_c], ignore_index=True)
#     merged_df['Algorithm'] = merged_df.groupby('Repository URL')['Algorithm'].transform(lambda x: ';'.join(x))
    
#     merged_df = merged_df.drop_duplicates(subset='Repository URL')
    
#     output_file_path = 'AD.csv'  
#     merged_df.to_csv(output_file_path, index=False)



if __name__ == "__main__":
    GITHUB_TOKEN=""
    libs = ["H2OIsolationForest", "LOF", "pyod.models.lof", "Anomalize", "pyod", "cuml"]

    for lib in libs:
        RunBash(GITHUB_TOKEN, lib)
        os.remove("output.txt")
        RemoveDuplicates("repository_stars.csv",lib)
    
    
    
    
    