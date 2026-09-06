import pandas as pd
from .encoding_constants import EncodingType
from .scaling_constants import ScalingType

def transform_encoding(encoding_type:EncodingType, x_train_str:pd.DataFrame, x_test_str:pd.DataFrame):

    assert encoding_type in EncodingType, "encoding_type값이 올바르지 않습니다."

    x_train_enc = pd.DataFrame()
    x_test_enc = pd.DataFrame()

    for col in x_train_str.columns:
        #학습
        encoding_type.value[1].fit(x_train_str[col].astype('category'))

        #변환
        encoded_col = encoding_type.value[1].transform(x_train_str[col].astype('category'))
        encoded_col_test = encoding_type.value[1].transform(x_test_str[col].astype('category'))

        x_train_enc = pd.concat([x_train_enc, encoded_col], axis=1)
        x_test_enc = pd.concat([x_test_enc, encoded_col_test], axis=1)

    return x_train_enc.reset_index(drop=True), x_test_enc.reset_index(drop=True)

def transform_scaling(scaling_type:ScalingType, x_train_number:pd.DataFrame, x_test_number:pd.DataFrame):

    assert scaling_type in ScalingType, "scaling_type값이 올바르지 않습니다."

    x_train_scaled = scaling_type.value[1].fit_transform(x_train_number)
    x_test_scaled = scaling_type.value[1].transform(x_test_number)

    x_train_scaled = pd.DataFrame(
        data = x_train_scaled, columns=x_train_number.columns)
    x_test_scaled = pd.DataFrame(
        data = x_test_scaled, columns=x_test_number.columns)

    return x_train_scaled.reset_index(drop=True), x_test_scaled.reset_index(drop=True)