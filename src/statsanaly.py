import json
import os

def calculate_stats(data, column):
    series = data[column].dropna()
    stats_dict = {
        'mean': float(series.mean()),
        'mode': float(series.mode().iloc[0]) if not series.mode().empty else None,
        'median': float(series.median()),
        'range': float(series.max() - series.min()),
        'std': float(series.std()),
        'variance': float(series.var()),
        'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else None,
        'quantile_25': float(series.quantile(0.25)),
        'quantile_50': float(series.quantile(0.50)),
        'quantile_75': float(series.quantile(0.75))
    }
    return stats_dict


def statsanaly(df):
    df['Family'] = ((df['Kidhome'] > 0) | (df['Teenhome'] > 0)).astype(int)



    family_1 = df[df['Family'] == 1]
    family_0 = df[df['Family'] == 0]

    results = {
        'family_1': {
            'Income': calculate_stats(family_1, 'Income'),
            'MntWines': calculate_stats(family_1, 'MntWines')
        },
        'family_0': {
            'Income': calculate_stats(family_0, 'Income'),
            'MntWines': calculate_stats(family_0, 'MntWines')
        }
    }

    with open(os.path.join('.', 'data', 'family_stats.json'), 'w') as f:
        json.dump(results, f, indent=4)


    print("\nDescriptive statistics done...")
    print("\nResults saved to file ./data/family_stats.json")