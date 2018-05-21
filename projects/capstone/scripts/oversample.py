from sklearn.utils import shuffle


def lop_off(df, decimal_pct):
    lop_off_idx = round(df.shape[0] * decimal_pct)
    final_test_set = df[:lop_off_idx]
    df = df[lop_off_idx:]
    return (df, final_test_set)


def oversample(df, col_name='stop_outcome', outcome_values=['Arrest', 'Summons', 'Ticket', 'Verbal Warning',
                                                            'Written Warning']):
    # tickets = df.loc[df['stop_outcome'] == 'Ticket']
    # arrests = df.loc[df['stop_outcome'] == 'Arrest']
    # summons = df.loc[df['stop_outcome'] == 'Summons']
    # ww = df.loc[df['stop_outcome'] == 'Written Warning']
    # vw = df.loc[df['stop_outcome'] == 'Verbal Warning']
    # multiplier_arrests = tickets.shape[0] // arrests.shape[0]
    # multiplier_summons = tickets.shape[0] // summons.shape[0]
    # multiplier_ww = tickets.shape[0] // ww.shape[0]
    # multiplier_vw = tickets.shape[0] // vw.shape[0]
    # oversampled = df.append([arrests] * multiplier_arrests, ignore_index=True)
    # oversampled = oversampled.append([summons] * multiplier_summons, ignore_index=True)
    # oversampled = oversampled.append([ww] * multiplier_ww, ignore_index=True)
    # oversampled = oversampled.append([vw] * multiplier_vw, ignore_index=True)

    print('Num rows before oversampling: {}'.format(df.shape[0]))
    counts = df[col_name].value_counts()
    largest = counts.nlargest(1)
    # Remove largest outcome value from 
    outcome_values.remove(largest.index[0])

    oversampled = df
    for val in outcome_values:
        multiplier = largest.iloc[0] // counts[val]
        print('multiplier for {}: {} row count: {}'.format(val, multiplier, oversampled.shape[0]))
        val_outcome_rows = oversampled.loc[oversampled[col_name] == val]
        print('num rows to be added: {}'.format(val_outcome_rows.shape[0] * multiplier))
        oversampled = oversampled.append([val_outcome_rows] * multiplier, ignore_index=True)
        print('num rows after added: {}'.format(oversampled.shape[0]))

    print('Num rows after oversampling: {}'.format(oversampled.shape[0]))

    return shuffle(oversampled, random_state=0)
