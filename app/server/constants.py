from enum import IntEnum


class TrainingStatus(IntEnum):
    CREATED = 1
    TRAINING_IN_PROGRESS = 2
    TRAINED = 3
    FAILED = 4