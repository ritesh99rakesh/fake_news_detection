import nltk
import pandas as pd


def clean_data(df, binary=True):
    print("============ Data Shape ============")
    print('Data shape: ', df.shape)
    df.columns = ['serialNo', 'ID', 'label', 'statement', 'subject', 'speaker', 'speakerTitle', 'state', 'party',
                  'barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'context', 'justification']
    del df['serialNo']
    del df['ID']
    pstemmer = nltk.stem.PorterStemmer()
    stopWords = set(nltk.corpus.stopwords.words('english'))

    values_nan = {'subject': 'unknown', 'speaker': 'unknown', 'speakerTitle': 'unknown', 'state': 'unknown',
                  'party': 'unknown', 'barely-true': 0.0, 'false': 0.0, 'half-true': 0.0, 'mostly-true': 0.0,
                  'pants-fire': 0.0, 'context': 'unknown', 'justification': 'unknown'}
    df.fillna(value=values_nan, inplace=True)

    if binary:
        df.label.replace(
            {'half-true': 1, 'mostly-true': 1, 'false': 0, 'true': 1, 'barely-true': 0, 'pants-fire': 0},
            inplace=True)
    else:
        df.label.replace(
            {'half-true': 2, 'mostly-true': 1, 'false': 4, 'true': 0, 'barely-true': 3, 'pants-fire': 5},
            inplace=True)

    df['statement'] = df['statement'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopWords]))
    df['statement'] = df['statement'].apply(lambda x: pstemmer.stem(x))

    df['context'] = df['context'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopWords]))
    df['context'] = df['context'].apply(lambda x: pstemmer.stem(x))

    df['justification'] = df['justification'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopWords]))
    df['justification'] = df['justification'].apply(lambda x: pstemmer.stem(x))

    df['speaker'] = df['speaker'].apply(lambda x: x.replace('-', ''))
    speaker_dict = dict()
    for i, speaker in enumerate(pd.unique(df['speaker'])):
        speaker_dict[speaker] = i

    df.speaker.replace(speaker_dict, inplace=True)

    speaker_title_dict = dict()
    for i, speakerTitle in enumerate(pd.unique(df['speakerTitle'])):
        speaker_title_dict[speakerTitle] = i

    df.speakerTitle.replace(speaker_title_dict, inplace=True)

    state_dict = dict()
    for i, state in enumerate(pd.unique(df['state'])):
        state_dict[state] = i

    df.state.replace(state_dict, inplace=True)

    party_dict = dict()
    for i, party in enumerate(pd.unique(df['party'])):
        party_dict[party] = i

    df.party.replace(party_dict, inplace=True)

    return df
