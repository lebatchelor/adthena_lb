from model_utils import *
import time


if __name__ == "__main__":

    if sys.argv[2] == 'train':
        t0 = time.time()
        processed_training_data = preprocess_data(file_path=sys.argv[1])
        print(f"Processing complete in {(time.time() - t0)}")
        X_train = processed_training_data['term_processed']
        y_train = processed_training_data[1]
        clf = train_model(X_train, y_train)
        print(f"Training complete in {(time.time() - t0)}")
        serialize(clf, filename='svc_model')

    elif sys.argv[2] == 'test':
        t0 = time.time()
        processed_predict_data = preprocess_data(file_path=sys.argv[1])
        print(f"Processing complete in {(time.time() - t0)}")
        X_test = processed_predict_data['term_processed']
        clf = deserialize('svc_model')
        y_pred = get_predictions(clf, X_test)
        print(f"Predictions complete in {(time.time() - t0)}")

        if len(processed_predict_data.columns) > 2:
            print(processed_predict_data.columns)
            y_test = processed_predict_data[1]
            get_test_stats(y_test, y_pred)
            print(f"Test stats complete in {(time.time() - t0)}")
        else:
            print('Model prediction complete')

    else:
        print('Please run again and specify parameter train or test in command.')


