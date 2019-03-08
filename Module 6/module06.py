import sys
import tokenize
from io import StringIO

def atom(next, token):
    if token[1] == '(':
        out = []
        token = next()
        while token[1] != ')':
            out.append(atom( next, token))
            token = next()
            if token[1] == ' ':
                token = next()
        return out
    elif token[1] == '?':
        token = next()
        return "?" + token[1]
    else:
        return token[1]

def parse(exp):
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())


def assert_parse(src, trg, debug):
    result = parse(src)
    if debug:
        print(src, result, trg)
    assert result == trg 


def test_parser(debug):
    assert_parse('Fred', 'Fred', debug)
    assert_parse('?x', '?x', debug)
    assert_parse("(loves Fred ?x)", ['loves', 'Fred', '?x'], debug)
    assert_parse("(father_of Barney (son_of Barney))", ['father_of', 'Barney', ['son_of', 'Barney']], debug)


def is_variable(exp):
    return isinstance(exp, str) and exp[0] == "?"


def is_constant(exp):
    return isinstance(exp, str) and not is_variable(exp)


def unification(list_expression1, list_expression2):
    ### YOUR SOLUTION HERE ###
    # implement the pseudocode in 2.3 of the assignment PDF
    ### YOUR SOLUTION HERE ### 
    pass


def unify(s_expression1, s_expression2):
    return unification(parse(s_expression1), parse(s_expression2))

unifications = 0
def assert_unify(exp1, exp2, trg, debug):
    # never do this for real code!
    global unifications
    result = unify(exp1, exp2)
    if debug:
        print(unifications, exp1, exp2, result, trg)
    assert result == trg 
    unifications += 1

def test_unify(debug):
    
    assert_unify('Fred', 'Barney', None, debug)
    # use underscores instead of hypthens
    assert_unify('(quarry_worker Fred)', '(quarry_worker ?x)', {"?x": 'Fred'}, debug)
    # add the remainder of the self check
    #
    # add 5 additional test cases.
    #

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    test_parser(debug)
    test_unify(debug)
    print(unifications) # should be 15
