import pandas as pd
 
def addrow(df, row):
    return df.append(pd.DataFrame(row), ignore_index=True)

def fare_in_bucket(fare, fare_bracket_size, bucket):
    return (fare > bucket * fare_bracket_size) & (fare <= ((bucket+1) * fare_bracket_size))

def build_survival_table(training_file):    
    fare_ceiling = 40
    train_df = pd.read_csv(training_file)
    train_df[train_df['Fare'] >= 39.0] = 39.0
    fare_bracket_size = 10
    number_of_price_brackets = fare_ceiling / fare_bracket_size
    number_of_classes = 3 #There were 1st, 2nd and 3rd classes on board 
     
    survival_table = pd.DataFrame(columns=['Sex', 'Pclass', 'PriceDist', 'Survived', 'NumberOfPeople'])
     
    for pclass in range(1, number_of_classes + 1): # add 1 to handle 0 start
        for bucket in range(0, number_of_price_brackets):
            for sex in ['female', 'male']:
                survival = train_df[(train_df['Sex'] == sex) 
                                    & (train_df['Pclass'] == pclass) 
                                    & fare_in_bucket(train_df["Fare"], fare_bracket_size, bucket)]
                            
                row = [dict(Pclass=pclass, Sex=sex, PriceDist = bucket, 
                            Survived = round(survival['Survived'].mean()), 
                            NumberOfPeople = survival.count()[0]) ]
                survival_table = addrow(survival_table, row)
     
    return survival_table.fillna(0)

def select_bucket(fare):
    if (fare >= 0 and fare < 10):
        return 0
    elif (fare >= 10 and fare < 20):
        return 1
    elif (fare >= 20 and fare < 30):
        return 2
    else:
        return 3

def calculate_survival(survival_table, row):
    survival_row = survival_table[(survival_table["Sex"] == row["Sex"]) & (survival_table["Pclass"] == row["Pclass"]) & (survival_table["PriceDist"] == select_bucket(row["Fare"]))]
    return int(survival_row["Survived"].iat[0])    

survival_table = build_survival_table("train.csv")
test_df = pd.read_csv('test.csv')             
test_df["Survived"] = test_df.apply(lambda row: calculate_survival(survival_table, row), axis=1)
test_df.to_csv("result.csv", cols=['PassengerId', 'Survived'], index=False)