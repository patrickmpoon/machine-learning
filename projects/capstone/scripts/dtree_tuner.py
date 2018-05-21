import humanize
import pickle
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


le_outcomes = LabelEncoder()

oversampled = pickle.load(open('../oversampled-labelencoded_test20180515105408.pkl', 'rb'))
oversampled['stop_outcome'] = le_outcomes.fit_transform(oversampled['stop_outcome'])

# Take out 5% of data for final final testing; shuffle first
oversampled = shuffle(oversampled, random_state=0)
outcomes = oversampled.pop('stop_outcome')
print("Num oversampled rows: {}".format(oversampled.shape[0]))

row_count = oversampled.shape[0]
lop_off_pct = .05
lop_off_idx = round(row_count * lop_off_pct)
print('lop_off_idx = {}'.format(lop_off_idx))

final_test_features = oversampled[:lop_off_idx]
final_test_outcomes = outcomes[:lop_off_idx]

subsample_pct = 1.0
subsample_idx = round(row_count * subsample_pct)
oversampled = oversampled[lop_off_idx:subsample_idx]
outcomes = outcomes[lop_off_idx:subsample_idx]

print("Num sample rows: {}".format(oversampled.shape[0]))

# # Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(oversampled, 
                                                    outcomes, 
                                                    test_size=0.2, 
                                                    random_state=0)


# n_est = 675 # 0.9312121844356352  3 mins
n_est = 700
rfc = RandomForestClassifier(verbose=3, random_state=0, n_jobs=8, n_estimators=n_est, max_features="sqrt", max_depth=None, criterion="entropy")

start = datetime.now()
rfc.fit(X_train, y_train)
end = datetime.now()


print('{}'.format(rfc.score(X_test, y_test)))

print(humanize.naturaldelta(end-start))
