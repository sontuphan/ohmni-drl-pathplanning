import sys
from env import test


if sys.argv[1] == '--test':
    if sys.argv[2] == 'env':
        test.show()

elif sys.argv[1] == '--ohmni':
    pass

else:
    print("Error: Invalid option!")
