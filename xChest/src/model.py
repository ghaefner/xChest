from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def split_train_data(dict_folder):
    print("[I] Splitting train data.")

    train_df, dummy_df = train_test_split(dict_folder['train'], train_size=0.8, shuffle= True, random_state= 42)
    valid_df, test_df= train_test_split(dummy_df, train_size= 0.6, shuffle= True, random_state= 42)

    return train_df, test_df, valid_df

