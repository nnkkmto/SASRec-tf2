
import numpy as np
import tensorflow as tf

from lib import preprocesser
from lib import data_handler
from model import sasrec


def build_tf_dataset(dataset_dict: dict, seq_max_len: int, behavior_col) -> [dict, np.array]:

    inputs = {}
    inputs['input_seq'] = tf.keras.preprocessing.sequence.pad_sequences(
        dataset_dict['behavior_' + behavior_col], padding='post', truncating='post', maxlen=seq_max_len)
    inputs['candidate'] = np.array(dataset_dict['candidate_' + behavior_col])

    output = np.array(dataset_dict['label'])

    return inputs, output


def main():
    seq_max_len = 150

    # load data
    filepath = 'data/unpacked/data.tsv'
    use_columns = ['user_id', 'product_name', 'order_number', 'add_to_cart_order']
    df = data_handler.load_tsv(filepath, use_columns)

    # encode
    encode_cols = ['product_name']
    df, ordinal_encoders = preprocesser.token_col_encode(df, encode_cols)

    # make dataset
    user_col = 'user_id'
    behavior_key_col = 'product_name'
    sort_cols = ['order_number', 'add_to_cart_order']
    dataset_dict = preprocesser.aggregate_features(df, user_col=user_col, sort_cols=sort_cols,
                                                   behavior_key_col=behavior_key_col,
                                                   behavior_category_cols=[],
                                                   seq_max_len=seq_max_len)

    # get counts
    item_num = len(ordinal_encoders['product_name'].get_params()['mapping'][0]['mapping'])

    # build dataset
    inputs, output = build_tf_dataset(dataset_dict, seq_max_len, behavior_cols)

    # build model
    model = sasrec(item_num=item_num, seq_max_len=seq_max_len)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.9)
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x=inputs, y=output, epochs=30, validation_split=0.2, batch_size=10000)


if __name__ == '__main__':
    main()





