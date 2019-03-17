import sys
from unification import parse, unify
from copy import deepcopy
from itertools import permutations
from collections import deque

def parse_list(alist):
    '''
    This routine parses the given list

    Args:
        alist - the list to parse
    Returns
        alist parsed
    '''
    return [parse(entry) for entry in alist]

def parse_dict(adict):
    '''
    This routine parses the given dictionary

    Args:
        adict - the dictionary to parse
    Returns
        adict parsed
    '''
    parsed = {}
    for key in adict.keys():
        parsed[key] = {}
        for sec_key in adict[key].keys():
            parsed[key][sec_key] = parse(adict[key][sec_key]) if isinstance(adict[key][sec_key], str) else parse_list(adict[key][sec_key])
    return parsed

def get_predicates(conditions, predicates):
    '''
    This routine gets all the predicates which
    matches the conditions

    Args:
        conditions - the preconditions
        predicates - the predicates of the state
    Return
        the predicates that applied to the condition
    '''
    flatten = [item for sublist in conditions for item in sublist]
    return [item for item in predicates if item[0] in flatten]

def split_predicates(predicates):
    '''
    This routine takes the predicates
    and splits them into a simple and complex lists.
    simple ones are of 'item Drill' complex ones
    are of 'at Me Home'

    Args:
        predicates - the predicates
    Returns:
        A simple and complex predicates list
    '''
    simple = []
    comp = []
    for item in predicates:
        if len(item) == 2:
            simple.append(deepcopy(item))
        else:
            comp.append(deepcopy(item))
    return simple, comp

def convert_list(item):
    '''
    This routine converts the given
    item to a formatted string
    
    Args:
        item - the list
    Return
        a formatted string
    '''
    try:
        str_item = ' '.join(item)
    except:
        return []
    return '({str_item})'.format(str_item=str_item)

def unify_expressions (exp1, exp2):
    '''
    This routine takes the given expressions,
    converts them to a string and unifies them

    Args:
        exp1 - the first expression
        exp2 - the second expression
    Returns
        the unification
    '''
    return unify(convert_list(exp1), convert_list(exp2))

def find_match(item, conditions, results):
    '''
    Thie routine find a match with the given item,
    conditions, and results

    Args:
        item - the item
        conditions - the conditions
        results - the results
    Returns:
        a conidition if found otherwise, None
    '''
    for y in range(len(conditions)):
        if item[0] == conditions[y][0] and conditions[y][1] not in results.keys():
            return conditions[y]
    return None

def compare_lists(lista, listb):
    '''
    This routine compares two lists

    Args:
        lista - the first list
        listb - the second list
    Returns:
        True if lista==listb
        False otherwise
    '''
    found = [True if x in lista else False for x in listb]
    if False in found or len(lista) != len(listb):
        return False
    else:
        return True

def is_in_list(item, alist):
    '''
    This routines determines whether the
    item is in the list of lists

    Args:
        item - a list
        alist - list of lists
    Return
        True if found, otherwise False
    '''
    for x in alist:
        if compare_lists(item, x[0]):
            return True
    return False

def is_in_dict(item, dictlist):
    '''
    This routines determines whether the
    item is in the list of dictionaries

    Args:
        item - a dictionary
        dictlist - list of dictionaries
    Return
        True if found, otherwise False
    '''
    found = [item == x for x in dictlist]
    if True in found:
        return True
    else:
        return False

def update_state(state, results, action):
    '''
    This routine creates a new state based
    on the results and action

    Args:
        state - the current state
        results - the results
        action - the action
    Returns:
        A new state with corresponding action
    '''
    new_state = deepcopy(state)
    delete = action['delete'][0]
    add = action['add'][0]
    action = action['action']
    del_from = [delete[x] if x == 0 else results[delete[x]] for x in range(len(delete))]
    add_to = [add[x] if x == 0 else results[add[x]] for x in range(len(add))]
    updated_action = [action[x] if x == 0 else results[action[x]] for x in range(len(action))]
    new_state.remove(del_from)
    new_state.append(add_to)
    return [new_state, updated_action]

def do_unifications(s_state, c_state, s_cond, c_cond, successors):
    '''
    This routine performs unification

    Args:
        s_state - simple states
        c_state - complex states
        s_cond - simple conditions
        c_cond - complex conditions
        successors - the successors
    Return:
        results or None
    '''
    res = {}
    # Iterate over the state and unify the simple predicates
    for x in s_state:
        test = find_match(x, s_cond, res)
        if test is not None:
            sample = unify_expressions(x, test)
            res = {**res, **sample}
    # If the sizes of the results are not equal
    # to the conditions then not all unifications
    # occured. Or if the results are already accounted
    # for, return None
    if len(res) < len(s_cond) or is_in_dict(res, successors):
        return None
    count = 0
    # Iterate over the state and unify the complex predicates
    for x in c_state:
        for y in c_cond:
            if x[0] == y[0]:
                if unify_expressions(x, [y[z] if z == 0 else res[y[z]] for z in range(len(y))]) == {}:
                    count += 1
    # If the count is less than the conditions then
    # the preconditions were not satisfied. Return None
    if count < len(c_cond):
        return None
    # If everything unified then return the simple predicates
    return res

def determine_successors(state, actions):
    '''
    This routine determines the successors for
    the given state

    Args:
        state - the state
        actions- the actions
    Returns:
        the successors
    '''
    successors = []
    # Iterare over all the actions
    for key in actions.keys():
        state_copy = deepcopy(state)
        # Get the predicates for the state
        predicates = get_predicates(actions[key]['conditions'], state_copy)
        # Split the predicates in the conditions into simple and complex
        s_cond, c_cond = split_predicates(actions[key]['conditions'])
        # Split the predicates into simple and complex ones
        simple, complx = split_predicates(predicates)
        # Obtain all the permutations of the simple list.
        # This allows to try all combinations in unification
        simple_combo = list(permutations(simple))
        temp = []
        # Iterate over the permutations and find the successor states
        for x in simple_combo:
            sucessor = do_unifications(x, complx, s_cond, c_cond, temp)
            if sucessor != None:
                temp.append(sucessor)
        for res in temp:
            successors.append(update_state(state_copy, res, actions[key]))
    return successors

def build_plan(plan, debug):
    '''
    This routine converts the plan into list string form
    and adjusts the indices

    Args:
        plan - the plan
        debug - the debug flag
    Returns:
        formatted plan
    '''
    def conversion(plan, index, is_goal=False):
        convlist = [convert_list(x) for x in plan[index][0]]
        if is_goal:
            return [convlist]  + ["Goal"]
        else:
            return [convlist] + [convert_list(plan[x+1][1])]
    formatted = []
    # Iterate over the plan minus the goal state
    for x in range(len(plan)-1):
        if debug:
            formatted.append(conversion(plan, x))
        else:
            formatted.append(convert_list(plan[x+1]))
    # When the debug flag is set add the goal
    # to the list
    if debug:
        formatted.append(conversion(plan, len(plan)-1, True))
    return formatted

def forward_planner( start_state, goal, actions, debug=False):
    '''
    The routine performs forward planning

    Args:
        start_state - the starting state
        goal - the goal state
        actions - the actions
        debug - debug flag
    Returns:
        when debug, list of [[state, actions], ..]
        otherwise, list of [[action] ..]
    '''
    start_state = parse_list(start_state)
    goal_state = parse_list(goal)
    actions = parse_dict(actions)
    frontier = deque([[start_state] + [None]])
    explored = []
    plan = []
    # Only loop if there is data in the frontier
    while len(frontier) >= 1:
        # Get the current state from the frontier
        current_state = frontier.pop()
        # When the debug flag is set add the
        # state and action to the plan otherwise,
        # just the action
        if debug:
            plan.append(current_state)
        else:
            plan.append(current_state[1])
        # If the current state is the goal state
        # return the plan
        print(current_state[0])
        if compare_lists(current_state[0], goal_state):
            return build_plan(plan, debug)
        # Determine the successors of the current state and
        # iterate over them. Only add to the frontier if the
        # successor is not in the frontier or not in the explored
        # list
        for successor in determine_successors(current_state[0], actions):
            if is_in_list(successor[0], explored) or is_in_list(successor[0], frontier):
                pass
            else:
                frontier.append(successor)
        # Add the current state to the frontier
        explored.append(current_state)
    return build_plan(plan, debug)

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    start_state = [
        "(item Drill)",
        "(place Home)",
        "(place Store)",
        "(agent Me)",
        "(at Me Home)",
        "(at Drill Store)",
    ]

    goal = [
        "(item Drill)",
        "(place Home)",
        "(place Store)",
        "(agent Me)",
        "(at Me Home)",
        "(at Drill Me)",
    ]

    actions = {
        "drive": {
            "action": "(drive ?agent ?from ?to)",
            "conditions": [
                "(agent ?agent)",
                "(place ?from)",
                "(place ?to)",
                "(at ?agent ?from)"
            ],
            "add": [
                "(at ?agent ?to)"
            ],
            "delete": [
                "(at ?agent ?from)"
            ]
        },
        "buy": {
            "action": "(buy ?purchaser ?seller ?item)",
            "conditions": [
                "(item ?item)",
                "(place ?seller)",
                "(agent ?purchaser)",
                "(at ?item ?seller)",
                "(at ?purchaser ?seller)"
            ],
            "add": [
                "(at ?item ?purchaser)"
            ],
            "delete": [
                "(at ?item ?seller)"
            ]
        }
    }
    # When debug is set:
    #   plan = [[state, action] [state, action] ..]
    # Otherwise:
    #   plan = [action, action, ..]
    plan = forward_planner( start_state, goal, actions, debug=debug)
    print(plan)


