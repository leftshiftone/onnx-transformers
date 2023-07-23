from enum import Enum
from typing import Union, List, Tuple

Text = str
TextPair = Tuple[Text, Text]

Batch = Union[Text, List[Text], TextPair, List[TextPair]]

Text = str
BatchText = List[Text]
Pair = Tuple[str, str]
BatchPair = List[Pair]
Input = Union[Text, BatchText, Pair, BatchPair]


class Direction(Enum):
    LEFT = "left"
    RIGHT = "right"
