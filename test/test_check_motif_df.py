import pandas as pd


def compare_dataframes(df, df_new):
    columns = df.columns
    differences = {}
    for column in columns:
        if not df[column].equals(df_new[column]):
            differences[column] = df[column].compare(df_new[column])

    if not differences:
        print("The dataframes are the same.")
    else:
        print("The dataframes are different.")
        print("Differences:")
        for column, diff in differences.items():
            print(f"Column: {column}")
            print(diff)


def main():
    # df1 = pd.read_json("data/raw-jsons/motifs/pdb_library_1_motifs.json")
    # df2 = pd.read_json("data/raw-jsons/motifs/pdb_library_1_motifs.new.json")
    df1 = pd.read_json("data/raw-jsons/constructs/pdb_library_1_combined.json.bak")
    df1.sort_values(
        by="name",
        key=lambda x: x.str.extract("(\d+)", expand=False).astype(int),
        inplace=True,
    )
    df1.reset_index(drop=True, inplace=True)
    df2 = pd.read_json("data/raw-jsons/constructs/pdb_library_1_combined.json")
    compare_dataframes(df1, df2)


if __name__ == "__main__":
    main()
