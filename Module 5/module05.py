import sys

# There is a way to do this assignment with higher order functions
# so that you only need one function to do the ga() and you delegate
# the brains of `binary_ga` and `real_ga` to it. 
#
# If this means nothing to you, ignore it. :D



def binary_ga(parameters, debug):
    ### YOUR SOLUTION HERE ###
    ### YOUR SOLUTION HERE ###
    pass

def real_ga(parameters, debug):
    ### YOUR SOLUTION HERE ###
    ### YOUR SOLUTION HERE ###
    pass

def shifted_sphere( shift, xs):
    return sum( [(x - shift)**2 for x in xs])

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    parameters = {
        "f": lambda xs: shifted_sphere( 0.5, xs),
        "minimization": True
        # put other parameters in here, add , to previous line.
    }
    print("Executing Binary GA")
    binary_ga(parameters, debug)

    print("Executing Real-Valued GA")
    parameters = {
        "f": lambda xs: shifted_sphere( 0.5, xs),
        "minimization": True
        # put other parameters in here, add , to previous line.
    }