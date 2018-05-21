# def oversample(df, col_name='stop_outcome', outcome_values=['Arrest', 'Summons', 'Ticket', 'Verbal Warning',
#                                                             'Written Warning']):
def oversample(df, col_name='stop_outcome', outcome_values=[0, 1, 2, 3, 4]):
    print('Num rows before oversampling: {}'.format(df.shape[0]))
    counts = df[col_name].value_counts()
    largest = counts.nlargest(1)

    # Remove largest outcome value from
    outcome_values.remove(largest.index[0])

    oversampled = df
    for val in outcome_values:
        # multiplier = largest.iloc[0] // counts[val]
        multiplier = 1
        if multiplier > 10:
            multiplier = 10
        print('multiplier for {}: {} row count: {}'.format(val, multiplier, oversampled.shape[0]))
        val_outcome_rows = oversampled.loc[oversampled[col_name] == val]
        print('num rows to be added: {}'.format(val_outcome_rows.shape[0] * multiplier))
        oversampled = oversampled.append([val_outcome_rows] * multiplier, ignore_index=True)
        print('num rows after added: {}'.format(oversampled.shape[0]))

    print('Num rows after oversampling: {}'.format(oversampled.shape[0]))

    return oversampled


def undersample(df, col_name='stop_outcome', outcome_values=[0, 1, 2, 3, 4]):
    dropoff_pct = 0.01
    col = df[col_name]
    counts = col.value_counts()
    largest = counts.nlargest(1).index[0]
    dropoff_idx = int(round(counts[largest] * dropoff_pct))

    indices = df[col == largest].index[:dropoff_idx]
    print('[Undersample] Row count before undersampling: {}'.format(df.shape[0]))
    print('[Undersample] count = {} dropoff_idx: {}'.format(counts[largest], dropoff_idx))
    df = df.drop(index=indices)
    print('[Undersample] Row count AFTER undersampling: {}'.format(df.shape[0]))
    return df
