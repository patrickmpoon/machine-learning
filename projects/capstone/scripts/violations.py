import numpy as np
import pandas as pd


violations = []


def normalize_violation(violation):
    """Normalize violation values
    """
    if violation == 'defective lights':
        return 'lights'
    elif violation == 'equipment violation':
        return 'equipment'
    elif violation == 'other/error':
        return 'other'
    elif violation == 'registration/plates':
        return 'registration'
    elif violation == 'seat belt':
        return 'seatbelt'
    elif violation == 'speed related':
        return 'speeding'
    elif violation == 'stop sign/light' or violation == 'stop sign':
        return 'bad_stop'
    return violation.replace(' ', '_')


def merge_violations(violations):
    """Merge violation and violation_raw columns
    """
    merged = []
    tokens = violations.lower().split(',')
    return list(set([normalize_violation(violation) for violation in tokens]))


def onehot_encode_violations(arr_violations):
    row = np.zeros(len(violations))
    for v in arr_violations:
        row[violations.index(v)] = 1
    return row


def normalize_violations(df):
    global violations

    for violation in list(df.violation.unique()) + list(df.violation_raw.unique()):
        tokens = violation.lower().split(',')
        violations.extend([normalize_violation(token) for token in tokens])

    violations = sorted(set(violations))

    merged = df[['violation_raw', 'violation']].apply(lambda x: ','.join(x), axis=1).apply(merge_violations)

    violation_col_headers = ['violation_{}'.format(violation.replace(' ', '_')) for violation in violations]

    df_violations = merged.apply(onehot_encode_violations).apply(lambda x: pd.Series(x, dtype=int))
    df_violations.columns = violation_col_headers

    return df_violations
