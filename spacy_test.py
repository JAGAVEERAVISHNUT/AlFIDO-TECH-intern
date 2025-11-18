import traceback
try:
    import spacy
    from spacytextblob.spacytextblob import SpacyTextBlob

    print('spacy version:', spacy.__version__)
    nlp = spacy.load('en_core_web_sm')
    print('pipeline before:', nlp.pipe_names)
    if 'spacytextblob' not in nlp.pipe_names:
        nlp.add_pipe('spacytextblob')
    print('pipeline after:', nlp.pipe_names)
    doc = nlp("I absolutely love this product!")
    print('polarity, subjectivity:', getattr(doc._, 'polarity', None), getattr(doc._, 'subjectivity', None))
except Exception:
    traceback.print_exc()
