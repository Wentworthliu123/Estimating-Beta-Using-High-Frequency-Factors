from main import tralpha
from main import tralpha_hf
from sys import argv

year = int(argv[1])
month = int(argv[2])
tralpha(year, month, 6)
tralpha_hf(year, month, 6)
