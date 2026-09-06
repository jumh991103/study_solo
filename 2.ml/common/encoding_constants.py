import enum
import category_encoders as ce

class EncodingType(enum.Enum):
    OneHot = (enum.auto(), ce.OneHotEncoder(use_cat_names=True))

