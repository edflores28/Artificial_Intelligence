import sys
import tokenize
from io import StringIO
from copy import deepcopy

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


def format_expression(exp):
    '''
    This routine formats the given expression

    Args:
        exp - the expression to format
    Reuturns:
        exp - formatted if type list, otherwise not formatted
    '''
    if isinstance(exp, list):
        return '(' + " ".join(exp) + ')'
    else:
        return exp

def is_variable_check(exp1, exp2):
    '''
    This routine does the variable processing
    of the unification algorithm

    Args:
        exp1 - the first expression
        exp2 - the second expression
    Reutrns:
        None when exp is in exp2
        {exp: exp2}, otherwise
    '''
    if exp1 in exp2:
        return None
    else:
        return {exp1: format_expression(exp2)}

def update_single(entry, key, value):
    '''
    This routine updates an entry from an expression
    with the given key and value

    Args:
        entry - the expression entry
        key - the key to replace
        value - the value of the key
    Reutrns:
        updated entry with value if key is present
    '''
    if len(entry) > 0 and isinstance(entry, str):
        return entry.replace(key, value)
    if len(entry) > 0 and isinstance(entry, list):
        return [value if x == key else x for x in entry]
    return entry

def update_entries(exp, key, value):
    '''
    This routine iterates over the expression
    and updates all entires
    
    Args:
        exp - the expression
        key - the key to replace
        value - the value of the key
    Reutrns:
        updated expression
    '''
    for x in range(len(exp)):
        exp[x] = update_single(exp[x], key, value)
    return exp

def update_expressions(result, exp1, exp2):
    '''
    This routine updates the expressions with
    the result

    Args:
        result - the unification result
        exp1 - the first expression
        exp2 - the second expression
    Returns:
        exp1 and exp2 updated
    '''
    exp1_s = deepcopy(exp1[1:])
    exp2_s = deepcopy(exp2[1:])
    # Only process if results is populated and iterate
    # over all keys and update the expressions
    if len(result) > 0:
        for key in result.keys():
            exp1_s = update_entries(exp1_s, key, result[key])
            exp2_s = update_entries(exp2_s, key, result[key])
    return exp1[:1] + exp1_s, exp2[:1] + exp2_s

def get_elements(exp1, exp2):
    '''
     This routine gets elements for both expressions

     Args:
        exp1 - the first expression
        exp2 - the second expression
    Returns:
        element1 and element2
    '''
    # Return and empty list is the expression is a
    # string. Otherwise get the first element
    # in the list
    element1 = [] if isinstance(exp1, str) else exp1[0]
    element2 = [] if isinstance(exp2, str) else exp2[0]
    return element1, element2 

def unification(list_expression1, list_expression2):
    '''
    The unification algorithm

    Args: 
        list_expression1 - the first expression
        list_expression2 - the second expression
    Returns:
        None unification fails 
        otherwise, a dictionary
    '''
    # Check if both expression are constants or empty lists
    if is_constant(list_expression1) and is_constant(list_expression2) or not list_expression1 or not list_expression2:
        if list_expression1 == list_expression2:
            return {}
        else:
            return None
    # Check if both expressions are variables
    if is_variable(list_expression1):
        return is_variable_check(list_expression1, list_expression2)
    if is_variable(list_expression2):
        return is_variable_check(list_expression2, list_expression1)
    # Obtain the first elements for both expressions
    first_1, first_2 = get_elements(list_expression1, list_expression2)
    # Call unification on the elements
    result1 = unification(first_1, first_2)
    if result1 is None:
        return None
    # Update both expressions based on the result
    list_expression1, list_expression2 = update_expressions(result1, list_expression1, list_expression2)
    # Call unification on the updated expressions
    result2 = unification(list_expression1[1:], list_expression2[1:])
    if result2 is None:
        return None
    # Reuturn the substitutions
    return {**result1, **result2}

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
    # Self check test cases
    assert_unify('Fred', 'Barney', None, debug)
    assert_unify('Pebbles', 'Pebbles', {}, debug)
    assert_unify('(quarry_worker Fred)', '(quarry_worker ?x)', {"?x": 'Fred'}, debug)
    assert_unify('(son Barney ?x)', '(son ?y Bam_Bam)', {"?x": 'Bam_Bam', "?y": 'Barney'}, debug)
    assert_unify('(married ?x ?y)', '(married Barney Wilma)', {"?x": 'Barney', "?y": 'Wilma'}, debug)
    assert_unify('(son Barney ?x)', '(son ?y (son Barney))', {"?y": 'Barney', "?x": "(son Barney)"}, debug)
    assert_unify('(son Barney ?x)', '(son ?y (son ?y))', {"?y": 'Barney', "?x": "(son Barney)"}, debug)
    assert_unify('(son Barney Bam_Bam)', '(son ?y (son Barney))', None, debug)
    assert_unify('(loves Fred Fred)', '(loves ?x ?x)', {"?x": 'Fred'}, debug)
    assert_unify('(future George Fred)', '(future ?y ?y)', None, debug)
    # Additional test cases
    assert_unify('(future George ?x Fred)', '(future ?y Bob ?z)', {"?y": 'George', "?x": 'Bob', "?z": 'Fred'}, debug)
    assert_unify('(son ?x (married ?x) ?y)', '(son (future ?z) ?w ?z)', {"?x": "(future ?z)", "?w": "(married (future ?z))", "?y": "?z" }, debug)
    assert_unify('(future ?x Fred ?z)', '(future Bob ?y married(?z))', None, debug)
    assert_unify('(future ?x Fred ?z)', '(future Bob ?y (married ?y ?x))', {'?x': 'Bob', '?y': 'Fred', '?z': '(married Fred Bob)'}, debug)
    assert_unify('(hates ?x)', '(future Bob)', None, debug)

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    test_parser(debug)
    test_unify(debug)
    print(unifications) # should be 15
