import os
from google.cloud import storage
storage_client = storage.Client()

def main():
    # give blob credentials
    workdir = os.getcwd()
    bucket_name = 'mishabot'
    bucket = storage_client.bucket(bucket_name)

    source_sbert_name= 'SBERT/bot/pytorch_model.bin'
    sbert_weights_path = os.path.join(workdir, 'SBERT/bot/pytorch_model.bin')

    source_cls_name= 'weights/classification.pt'
    sbert_cls_path = os.path.join(workdir, 'weights/classification.pt')

    models = [[source_sbert_name, sbert_weights_path], \
        [source_cls_name, sbert_cls_path]]

    # get bucket object
    for i, model in enumerate(models):
        if not os.path.exists(model[1]):
            try:
                blob = bucket.blob(model[0])
                blob.download_to_filename(model[1])
                print('file: ', model[0], \
                    ' downloaded from bucket: ',bucket_name,' successfully')
            except Exception as e:
                print(e)
        else:
            if i == 0:
                print('Sentence BERT model exists')
            else: 
                print('Classifier model exists') 


if __name__ == '__main__':
    main()