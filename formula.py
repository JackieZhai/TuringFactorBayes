from utils import *


def tree_to_formula(tree, for_print=False):
    if isinstance(tree, node):
        if tree.name in ['+', '-', '*', '/', '>', '<', '==', '>=', '<=', '^','||','&']:
            string_1 = tree_to_formula(tree.children[0], for_print)
            string_2 = tree_to_formula(tree.children[1], for_print)
            return '(' + string_1 + tree.name + string_2 + ')'
        elif tree.name == '?':
            result = ['((', tree_to_formula(tree.children[0], for_print), ')?(', 
                      tree_to_formula(tree.children[1], for_print), '):(',
                      tree_to_formula(tree.children[2], for_print), '))']
            return ''.join(result)
        else:
            if for_print:
                result = [bcolors.OP_NODE + tree.name + bcolors.END, '(']
            else:
                result = [tree.name, '(']
            for i in range(len(tree.children)):
                string_i = tree_to_formula(tree.children[i], for_print)
                result.append(string_i)
                result.append(',')
            result.pop()
            result.append(')')
            return ''.join(result)
    elif isinstance(tree, paramnode):
        if for_print:
            return bcolors.VAR_NODE + str(tree.name) + bcolors.END
        else:
            return str(tree.name)
    elif isinstance(tree, constnode):
        if for_print:
            if tree.isTune:
                return bcolors.CONST_NODE_FOR_TUNE + str(tree.name) + bcolors.END
            else:
                return bcolors.CONST_NODE + str(tree.name) + bcolors.END
        else:
            return str(tree.name)

def check_sanity(formula):
    count = 0
    for i in range(len(formula)):
        if formula[i] == '(':
            count += 1
        if formula[i] == ')':
            count -= 1
        if count < 0:
            return 0
    if count != 0:
        return 0
    return 1

def formula_to_tree(formula):

    formula = formula.replace(' ','')
    formula = formula.replace('*-1', '*(0-1)')
    formula = formula.replace('(-1', '((0-1)')

    while(1):
        if formula[0] == '(':
            if check_sanity(formula[1:-1]):
                formula = formula[1:-1]
            else:
                break
        else:
            break
    if formula in var_names:
        return paramnode(var_names.index(formula))
    elif formula.isdigit():
        return constnode(int(formula))
    elif formula[0] == '0' and formula[1] == '.' and len(formula) < 5:
        return constnode(float(formula))
    var_stack = []
    op_stack = []
    count = 0
    pointer = 0
    k = 0
    while k < len(formula):
        if formula[k] == '(' and count == 0:
            for i in range(k, len(formula)):
                if formula[i] == '(': count += 1
                elif formula[i] == ')': count -= 1
                if count == 0: break
            index = i  # index points at ')'
            tree_k = formula_to_tree(formula[k:index+1])
            var_stack.append(tree_k)
            if i == len(formula) - 1:
                k = i
                pointer = i
            else:
                k = i
                pointer = i+1
        if formula[k] in ['+', '-', '*', '/', '?', '>','<', '&', '^']:
            op = formula[k]
            for func in func_list:
                if func.name == op:
                    fw_k = func
                    break
            op_stack.append(fw_k)
            pointer = k+1
        elif formula[k] == ':':
            pointer = k+1
            count = 0
            flag = 0
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    count += 1
                    flag = 1
                elif formula[i] == ')':
                    count -= 1
                if count == 0 and flag == 1:
                    # i points at ')corn --workers=3  xshow:app -b 0.0.0.0:8080 &'
                    var_stack.append(formula_to_tree(formula[pointer:i+1]))
                    if i+1 == len(formula):
                        pointer = i
                    else:
                        pointer = i+1
                    k = pointer
                    break
                elif count < 0 and flag == 0:
                    var_stack.append(formula_to_tree(formula[pointer:i]))
                    pointer = i
                    break
        elif formula[pointer:k+1] in unary_list:
            cnt_tem = 0
            op = formula[pointer:k+1]
            for func in func_list:
                if func.name == op:
                    fw = func
                    break
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    cnt_tem += 1
                if formula[i] == ')':
                    cnt_tem -= 1
                if cnt_tem == 0:
                    children = [formula_to_tree(formula[k+1:i+1])]
                    tree_k = node(fw, children)
                    var_stack.append(tree_k)
                    break
            k = i
            pointer = k+1
        elif formula[pointer:k+1] in ['==', '<', '>=', '<=', '||']:
            op = formula[pointer:k+1]
            for func in func_list:
                if func.name == op:
                    fw_k = func
                    break
            op_stack.append(fw_k)
            pointer = k+1
        elif (formula[pointer:k+1] in ts_list or formula[pointer:k+1] in binary_list) and formula[k+1] == '(':
            children = []
            cnt_tem = 0
            op = formula[pointer:k+1]
            for func in func_list:
                if func.name == op:
                    fw = func
                    break
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    cnt_tem += 1
                if formula[i] == ')':
                    cnt_tem -= 1
                if formula[i] == ',' and cnt_tem == 1:
                    index = i
                if cnt_tem == 0:
                    children.append(formula_to_tree(formula[k+2:index]))
                    children.append(formula_to_tree(formula[index+1:i]))
                    tree_k = node(fw, children)
                    var_stack.append(tree_k)
                    break
            k = i
            pointer = k+1
        elif formula[pointer:k+1] in bi_ts_list:
            cnt_tem = 0
            idx_list = []
            children = []
            op = formula[pointer:k+1]
            for func in func_list:
                if func.name == op:
                    fw = func
                    break
            for i in range(k+1, len(formula)):
                if formula[i] == '(':
                    cnt_tem += 1
                if formula[i] == ')':
                    cnt_tem -= 1
                if formula[i] == ',' and cnt_tem == 1:
                    idx_i = i
                    idx_list.append(idx_i)
                if cnt_tem == 0:
                    children.append(formula_to_tree(formula[k+2:idx_list[0]]))
                    children.append(formula_to_tree(formula[idx_list[0]+1:idx_list[1]]))
                    children.append(formula_to_tree(formula[idx_list[1]+1:i+1]))
                    tree_k = node(fw, children)
                    var_stack.append(tree_k)
                    break
            k = i
            pointer = k+1
        elif formula[pointer:k+1] in var_names and (k+1 == len(formula) or not formula[k+1].isalpha()):
            var_stack.append(paramnode(var_names.index(formula[pointer:k+1])))
            pointer = k+1
        elif formula[k].isdigit() and (k+1 == len(formula) or formula[k+1] not in '.0123456789'):
            if(k==pointer):
                var_stack.append(constnode(int(formula[pointer:k+1])))
                pointer = k+1
            else:
                var_stack.append(constnode(float(formula[pointer:k+1]))) # C:int(float(formula...))
                pointer = k+1
        elif formula[pointer] == '0' and (k+1 == len(formula) or formula[k+1] not in '.0123456789'):
            var_stack.append(constnode(float(formula[pointer:k+1])))
            pointer = k+1
        # elif '.' in formula[pointer:k+1] and (k+1 == len(formula) or formula[k+1] not in '0123456789'):
        #     var_stack.append(constnode)
        k += 1
        if len(op_stack) > 0:
            while op_stack[-1].name == '?':
                if len(var_stack) < 3:
                    break
                else:
                    fw = op_stack.pop()
                    rchild = var_stack.pop()
                    mchild = var_stack.pop()
                    lchild = var_stack.pop()
                    children = [lchild, mchild, rchild]
                    tree_k = node(fw, children)
                    var_stack.append(tree_k)
                if len(op_stack) == 0:
                    break
            while len(var_stack) >= 2 and op_stack[-1].name != '?':
                try:
                    fw = op_stack.pop()
                    rchild = var_stack.pop()
                    lchild = var_stack.pop()
                    children = [lchild, rchild]
                    tree_k = node(fw, children)  # k here is to avoid duplicate
                    var_stack.append(tree_k)
                except:
                    for item in var_stack:
                        if isinstance(item, constnode):print(item.name)
                        else:print(item.name)
                    raise NotImplementedError
    # if len(var_stack) != 1 or len(op_stack) != 0:
    #     print("There's an error!")
    #     raise NotImplementedError
    return var_stack[0]

if __name__ == '__main__':
    print(var_names)
    formula = 'regbeta(mean(close,6),sequence(6),6)'
    tree = formula_to_tree(formula)
    print(tree.display())
    formula = tree_to_formula(tree)
    print(formula)