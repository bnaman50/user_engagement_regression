import pandas as pd
import numpy as np
import settings
import os
import sys

## TODO:
'''
 As of now, I load the file directly for train, val, test split. 
In future, you can read in chunks and then save it.
'''

def main(seed):
    df = pd.read_csv(settings.file_name)
    train, val, test = np.split(df.sample(frac=1, random_state=seed),
                                      [int(.5 * len(df)), int(.75 * len(df))],
                                      )

    if sys.gettrace() is not None:
        print(f'Not saving the files in debugging mode')
    else:
        print(f'Saving')
        train.to_csv(os.path.join(settings.data_dir, 'train.csv'), index=False)
        val.to_csv(os.path.join(settings.data_dir, 'val.csv'), index=False)
        test.to_csv(os.path.join(settings.data_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    seed=40
    main(seed)
    print(f'Done')