import enum
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ScalingType(enum.Enum):
    Standard = (enum.auto(), StandardScaler())

